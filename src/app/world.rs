use glam::Vec3;
use wgpu::util::DeviceExt;

use hex3::geometry::{MeshVertex, SurfaceVertex, UnifiedMesh, VoronoiMesh};
use hex3::render::{create_index_buffer, create_vertex_buffer, ElevationVertex};
use hex3::util::Timed;
use hex3::world::{FineCacheMode, VoronoiBackend, World};

use super::coloring::{
    cell_color_climate, cell_color_elevation, cell_color_feature, cell_color_hydrology,
    cell_color_noise, cell_color_plate, cell_color_terrain, cell_material,
};
use super::view::{ClimateLayer, FeatureLayer, NoiseLayer, RenderMode};
use super::visualization::{
    build_boundary_edge_colors, generate_pole_markers, generate_velocity_arrows,
};

pub const NUM_CELLS: usize = 100000;
/// Parity-only: the authoritative Lloyd relaxation count currently lives inside
/// `Tessellation`; this value keeps app logging/call sites honest.
pub const LLOYD_ITERATIONS: usize = 2;
pub const NUM_PLATES: usize = hex3::world::NUM_PLATES_DEFAULT;

/// Minimum flow for "all rivers" mode, as fraction of total cells.
/// E.g., 0.00005 means a cell needs 0.005% of total cells draining through it.
/// Lowered 0.0003→0.00005 (river-render): the draped texture can afford a much denser,
/// more dendritic network than the fat quads could; shows tributaries, not just trunks.
const RIVER_MIN_FLOW_FRACTION: f32 = 0.00005;

/// Minimum flow for a river mouth to be a "major outlet", as fraction of total cells.
/// Rivers are traced upstream from outlets exceeding this threshold.
/// E.g., 0.002 means outlet must drain 0.2% of all cells.
const RIVER_OUTLET_FRACTION: f32 = 0.004;

/// Minimum flow to continue tracing a river branch, as fraction of total cells.
/// At confluences, branches above this threshold are followed.
const RIVER_BRANCH_FRACTION: f32 = 0.0006;

fn buffer_bytes<T>(items: &[T]) -> usize {
    items.len() * std::mem::size_of::<T>()
}

/// Relief-mode wireframe: an indexed line list of the Voronoi edges, elevation
/// displaced. Built on demand (see [`generate_relief_edge_buffers`]) rather than
/// inside [`generate_world_buffers`].
pub struct ReliefEdgeBuffers {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

/// All GPU buffers for world rendering.
/// Simplified: one dynamic colored mesh + specialized buffers for Relief/rivers/overlays.
pub struct WorldBuffers {
    // Dynamic colored mesh (regenerated on mode/layer switch) - used by most modes
    pub colored_vertex_buffer: wgpu::Buffer,
    pub colored_index_buffer: wgpu::Buffer,
    pub num_colored_indices: u32,

    // Edge lines (two variants: default gray, plates with colored boundaries)
    pub edge_vertex_buffer: wgpu::Buffer,
    pub edge_vertex_buffer_plates: wgpu::Buffer,
    pub num_edge_vertices: u32,
    pub num_edge_vertices_plates: u32,

    // Relief mode: unified mesh with materials + elevation + relief edges
    pub unified_vertex_buffer: wgpu::Buffer,
    pub unified_index_buffer: wgpu::Buffer,
    pub num_unified_indices: u32,
    // Relief wireframe: built lazily the first time edges are shown, so its
    // large allocation (hundreds of MiB at the 2.5M-cell fine mesh) never
    // coincides with the fill-mesh rebuild. `None` after every buffer regen.
    pub relief_edge: Option<ReliefEdgeBuffers>,

    // Line-based rivers (non-relief modes). Relief mode uses the draped SDF texture below.
    pub river_all_vertex_buffer: wgpu::Buffer,
    pub river_major_vertex_buffer: wgpu::Buffer,
    pub num_river_all_vertices: u32,
    pub num_river_major_vertices: u32,

    // Draped rivers: baked equirect river texture (group 1) + its bind group. Always
    // present (transparent when there are no rivers / pre-hydrology) so the unified
    // pipeline's group 1 is always bindable. `_river_texture` is kept alive for the view.
    pub river_bind_group: wgpu::BindGroup,
    _river_texture: wgpu::Texture,

    // Plate overlays (arrows + pole markers)
    pub arrow_vertex_buffer: wgpu::Buffer,
    pub pole_marker_vertex_buffer: wgpu::Buffer,
    pub pole_marker_index_buffer: wgpu::Buffer,
    pub num_arrow_vertices: u32,
    pub num_pole_marker_indices: u32,
}

impl WorldBuffers {
    /// Get the river buffer and vertex count for the given river mode.
    pub fn river_buffer(&self, mode: super::view::RiverMode) -> Option<(&wgpu::Buffer, u32)> {
        match mode {
            super::view::RiverMode::Off => None,
            super::view::RiverMode::Major => Some((
                &self.river_major_vertex_buffer,
                self.num_river_major_vertices,
            )),
            super::view::RiverMode::All => {
                Some((&self.river_all_vertex_buffer, self.num_river_all_vertices))
            }
        }
    }
}

/// Runtime erosion-knob overrides from the CLI, applied to a freshly created
/// world before its stages are computed. `None` keeps the `EROSION_*` default.
/// Lets the interactive app A/B the routing-ladder rungs (e.g. MFD incision) on
/// Windows without a recompile. See docs/specs/erosion-routing-ladder.md.
#[derive(Clone, Copy, Default)]
pub struct ErosionOverrides {
    pub mfd_exponent: Option<f32>,
    pub flat_resolution: Option<bool>,
    pub confinement_slope: Option<f32>,
    pub k: Option<f32>,
    pub n: Option<f32>,
    pub diffusivity: Option<f32>,
    pub channel_support_km2: Option<f32>,
    pub uplift_smooth_km: Option<f32>,
    pub hillslope_critical_slope: Option<f32>,
    pub diffusion_iters: Option<usize>,
    pub reroute_interval: Option<usize>,
    pub steps: Option<usize>,
    pub precip_outer_iters: Option<usize>,
    pub uplift_scale: Option<f32>,
    pub rebuild_gain: Option<f32>,
    pub deposition_slope: Option<f32>,
    pub litho_sigma: Option<f32>,
    pub litho_grain_strength: Option<f32>,
    pub orographic_precip_strength: Option<f32>,
    pub downwind_shadow_strength: Option<f32>,
    pub lake_evap_strength: Option<f32>,
    /// Climate ratio (precip/evaporation). NOTE: consumed post-generation by the sweep
    /// (`set_active_climate_ratio`), not in `apply` — adjusts lake levels on the drainage.
    pub climate_ratio: Option<f32>,
    pub glacial_k: Option<f32>,
    // A4 drainage pulse (meso-a4-drainage-pulse.md): erosion-side two-stage
    // burn-in → trunk/interfluve uplift modifier → frozen final epoch.
    pub drainage_pulse: Option<f32>,
    pub pulse_burnin_steps: Option<usize>,
    pub pulse_smooth_km: Option<f32>,
    // Fine-base structural-relief knobs (P1a): these target `fine_structure_params`,
    // NOT `erosion_params` — they shape the pre-erosion base, so `apply` must run
    // before stage-3 fine generation (it does in every path here). Decision A.
    pub fault_scarp_height: Option<f32>,
    pub interior_relief: Option<f32>,
    pub front_strike_weight: Option<f32>,
    pub margin_contrast: Option<f32>,
    pub emergent_lambda: Option<f32>,
    pub emergent_structured: Option<f32>,
    pub meso_relief: Option<f32>,
    pub meso_irregularity: Option<f32>,
    pub meso_style: Option<usize>,
    pub meso_base_relief: Option<f32>,
    pub meso_wavelength_km: Option<f32>,
}

impl ErosionOverrides {
    pub fn apply(&self, world: &mut World) {
        if let Some(p) = self.mfd_exponent {
            world.erosion_params.mfd_exponent = p;
        }
        if let Some(f) = self.flat_resolution {
            world.erosion_params.flat_resolution = f;
        }
        if let Some(s) = self.confinement_slope {
            world.erosion_params.confinement_slope = s;
        }
        if let Some(k) = self.k {
            world.erosion_params.k = k;
        }
        if let Some(n) = self.n {
            world.erosion_params.n = n;
        }
        if let Some(d) = self.diffusivity {
            world.erosion_params.diffusivity = d;
        }
        if let Some(c) = self.channel_support_km2 {
            world.erosion_params.channel_support_km2 = c;
        }
        if let Some(s) = self.uplift_smooth_km {
            world.erosion_params.uplift_smooth_km = s;
        }
        if let Some(sc) = self.hillslope_critical_slope {
            world.erosion_params.hillslope_critical_slope = sc;
        }
        if let Some(i) = self.diffusion_iters {
            world.erosion_params.diffusion_iters = i;
        }
        if let Some(r) = self.reroute_interval {
            world.erosion_params.reroute_interval = r;
        }
        if let Some(n) = self.steps {
            world.erosion_params.steps = n;
        }
        if let Some(p) = self.precip_outer_iters {
            world.erosion_params.precip_outer_iters = p;
        }
        if let Some(u) = self.uplift_scale {
            world.erosion_params.uplift_scale = u;
        }
        if let Some(g) = self.rebuild_gain {
            world.erosion_params.rebuild_gain = g;
        }
        if let Some(d) = self.deposition_slope {
            world.erosion_params.deposition_slope = d;
        }
        if let Some(s) = self.litho_sigma {
            world.erosion_params.litho_sigma = s;
        }
        if let Some(g) = self.litho_grain_strength {
            world.erosion_params.litho_grain_strength = g;
        }
        if let Some(o) = self.orographic_precip_strength {
            world.erosion_params.orographic_precip_strength = o;
        }
        if let Some(d) = self.downwind_shadow_strength {
            world.erosion_params.downwind_shadow_strength = d;
        }
        if let Some(l) = self.lake_evap_strength {
            world.erosion_params.lake_evap_strength = l;
        }
        if let Some(g) = self.glacial_k {
            world.erosion_params.glacial_k = g;
        }
        if let Some(p) = self.drainage_pulse {
            world.erosion_params.drainage_pulse = p;
        }
        if let Some(s) = self.pulse_burnin_steps {
            world.erosion_params.pulse_burnin_steps = s;
        }
        if let Some(s) = self.pulse_smooth_km {
            world.erosion_params.pulse_smooth_km = s;
        }
        if let Some(f) = self.fault_scarp_height {
            world.fine_structure_params.fault_scarp_height = f;
        }
        if let Some(r) = self.interior_relief {
            world.fine_structure_params.interior_relief = r;
        }
        if let Some(w) = self.front_strike_weight {
            world.fine_structure_params.front_strike_weight = w;
        }
        if let Some(m) = self.margin_contrast {
            world.fine_structure_params.margin_contrast = m;
        }
        if let Some(l) = self.emergent_lambda {
            world.fine_structure_params.emergent_lambda = l;
        }
        if let Some(s) = self.emergent_structured {
            world.fine_structure_params.emergent_structured = s;
        }
        if let Some(r) = self.meso_relief {
            world.fine_structure_params.meso_relief = r;
        }
        if let Some(g) = self.meso_irregularity {
            world.fine_structure_params.meso_irregularity = g;
        }
        if let Some(st) = self.meso_style {
            world.fine_structure_params.meso_style = st;
        }
        if let Some(r) = self.meso_base_relief {
            world.fine_structure_params.meso_base_relief = r;
        }
        if let Some(w) = self.meso_wavelength_km {
            world.fine_structure_params.meso_wavelength_km = w;
        }
    }
}

pub fn create_world_with_options(
    seed: u64,
    num_cells: usize,
    backend: VoronoiBackend,
    fine_cache: FineCacheMode,
) -> World {
    let _total = Timed::info("Stage 1 (Lithosphere)");
    log::info!(
        "Generating world: seed={}, cells={}, lloyd={}, plates={}, voronoi_backend={}",
        seed,
        num_cells,
        LLOYD_ITERATIONS,
        NUM_PLATES,
        backend
    );

    let mut world = {
        let _t = Timed::info("Tessellation");
        World::new_with_options(seed, num_cells, LLOYD_ITERATIONS, backend)
    };
    world.fine_cache = fine_cache;

    {
        let _t = Timed::info("Plates");
        world.generate_plates(NUM_PLATES);
    }

    {
        let _t = Timed::info("Crust");
        world.generate_crust();
    }

    {
        let _t = Timed::info("Dynamics");
        world.generate_dynamics();
    }

    {
        let _t = Timed::info("Features");
        world.generate_features();
    }

    {
        let _t = Timed::info("Elevation");
        world.generate_elevation();
    }

    print_world_stats(&world);

    world
}

/// Advance world to Stage 2 (Atmosphere).
pub fn advance_to_stage_2(world: &mut World) {
    let _total = Timed::info("Stage 2 (Atmosphere)");

    {
        let _t = Timed::info("Atmosphere");
        world.generate_atmosphere();
    }

    // Print atmosphere stats
    if let Some(atmosphere) = &world.atmosphere {
        let stats = atmosphere.stats();
        let (mean_wind_delta, max_wind_delta, mean_upper, mean_surface) = {
            let n = atmosphere.wind.len().max(1) as f32;
            let mut sum_delta = 0.0_f32;
            let mut max_delta = 0.0_f32;
            let mut sum_upper = 0.0_f32;
            let mut sum_surface = 0.0_f32;
            for (u, s) in atmosphere.upper_wind.iter().zip(atmosphere.wind.iter()) {
                sum_upper += u.length();
                sum_surface += s.length();
                let d = (*u - *s).length();
                sum_delta += d;
                max_delta = max_delta.max(d);
            }
            (sum_delta / n, max_delta, sum_upper / n, sum_surface / n)
        };
        log::info!(
            "Atmosphere: temp=[{:.2}, {:.2}], mean={:.2}, mean_wind=[{:.2},{:.2}], max_wind=[{:.2},{:.2}], wind_delta=[{:.3},{:.3}], max_uplift={:.2}",
            stats.min_temp,
            stats.max_temp,
            stats.mean_temp,
            mean_upper,
            mean_surface,
            stats.max_upper_wind,
            stats.max_wind,
            mean_wind_delta,
            max_wind_delta,
            stats.max_uplift
        );
    }
}

/// Advance world to Stage 3 (Hydrosphere): fine mesh + hydrology on the
/// PRE-erosion terrain. Erosion is stage 4 ([`advance_to_stage_4`]).
pub fn advance_to_stage_3(world: &mut World) {
    let _total = Timed::info("Stage 3 (Hydrosphere)");
    {
        let _t = Timed::info("Fine mesh + pre-erosion hydrology");
        world.generate_fine_pre();
    }
    log_hydrology_stats(world);
}

/// Advance world to Stage 4 (Erosion): carve the fine mesh and re-derive
/// hydrology over the eroded terrain. Requires stage 3.
pub fn advance_to_stage_4(world: &mut World) {
    let _total = Timed::info("Stage 4 (Erosion)");
    {
        let _t = Timed::info("Erosion + hydrology");
        world.generate_fine_eroded();
    }
    log_hydrology_stats(world);
}

/// Log hydrology/fine-mesh stats for whatever stage is currently active.
fn log_hydrology_stats(world: &World) {
    let Some(hydrology) = world.active_hydrology() else {
        return;
    };
    let num_cells = world.num_cells();

    let ocean_cells = (0..num_cells).filter(|&i| hydrology.is_ocean(i)).count();
    let lake_cells = (0..num_cells)
        .filter(|&i| hydrology.is_lake_water(i))
        .count();
    let dry_basin_cells = (0..num_cells)
        .filter(|&i| hydrology.is_dry_basin(i))
        .count();
    let land_cells = (0..num_cells)
        .filter(|&i| !hydrology.is_submerged(i))
        .count();
    let non_ocean_cells = num_cells - ocean_cells;
    let lake_pct = if non_ocean_cells > 0 {
        100.0 * lake_cells as f32 / non_ocean_cells as f32
    } else {
        0.0
    };

    let cells_with_drainage = (0..num_cells)
        .filter(|&i| hydrology.downstream(i).is_some())
        .count();

    let river_min_flow = (num_cells as f32 * RIVER_MIN_FLOW_FRACTION).max(1.0);
    let river_cells = hydrology.river_cells(river_min_flow);
    // Count-equivalent so the logged figure stays comparable to river_min_flow
    // (both in upstream-cell units) regardless of mesh.
    let max_flow = (0..num_cells)
        .map(|i| hydrology.flow_count_equiv(i))
        .fold(0.0f32, f32::max);

    log::info!(
        "Hydrology: ocean={}, land={}, lakes={} ({:.1}%), basins={} ({} dry)",
        ocean_cells,
        land_cells,
        lake_cells,
        lake_pct,
        hydrology.basins.len(),
        dry_basin_cells
    );
    log::info!(
        "Rivers: drainage={} cells, rivers={} (flow>={:.0}), max_flow={:.0}",
        cells_with_drainage,
        river_cells.len(),
        river_min_flow,
        max_flow
    );
    if let Some(fine) = &world.fine {
        log::info!(
            "Fine mesh: coarse_cells={}, fine_cells={}, density_ratio={:.1}:1",
            world.tessellation.num_cells(),
            fine.tessellation().num_cells(),
            fine.achieved_density_ratio()
        );
    }
}

/// Compute resolution-independent river thresholds from cell count.
/// River rendering color - muted blue matching lake/water colors.
const RIVER_COLOR: Vec3 = Vec3::new(0.15, 0.35, 0.60);

/// Which subset of river segments to render.
#[derive(Clone, Copy)]
enum RiverSet {
    /// All drainage above the minimum flow threshold.
    All,
    /// Only cells belonging to major rivers (traced upstream from large outlets).
    Major,
}

struct RiverRenderData {
    include_all: Vec<bool>,
    include_major: Vec<bool>,
    max_flow: f32,
    log_max_flow_count: f32,
    lake_outflow_paths: Vec<(usize, Vec<usize>)>,
}

impl RiverRenderData {
    fn include(&self, set: RiverSet) -> &[bool] {
        match set {
            RiverSet::All => &self.include_all,
            RiverSet::Major => &self.include_major,
        }
    }
}

/// River render threshold calibration. `Catchment` (the default) thresholds on
/// PHYSICAL catchment area (km² at land-mean wetness) — resolution- and
/// adaptive-mesh-independent. `Legacy` is the old count-equivalent scheme,
/// kept for A/B: it was tuned on the coarse mesh, and on the adaptive fine
/// mesh its effective catchment is enormous (the "stub rivers" defect measured
/// by `diagnose --river-audit`). Select with `--river-legacy` /
/// `--river-min-catchment-km2`.
#[derive(Clone, Copy, Debug)]
pub enum RiverThresholdMode {
    Legacy,
    /// Minimum catchment (km²) for the 'All' network; Major outlet/branch
    /// scale with it (75× / 12.5×).
    CatchmentKm2(f32),
}

/// Default minimum catchment (km²) that renders as a river. Earth topographic
/// maps at global scale show perennial rivers from roughly 1-5k km² catchments.
pub const RIVER_DEFAULT_MIN_CATCHMENT_KM2: f32 = 2000.0;

static RIVER_THRESHOLD_MODE: std::sync::OnceLock<RiverThresholdMode> = std::sync::OnceLock::new();

/// Set once at startup (before any world buffers are built).
pub fn set_river_threshold_mode(mode: RiverThresholdMode) {
    let _ = RIVER_THRESHOLD_MODE.set(mode);
}

fn river_threshold_mode() -> RiverThresholdMode {
    *RIVER_THRESHOLD_MODE
        .get()
        .unwrap_or(&RiverThresholdMode::CatchmentKm2(
            RIVER_DEFAULT_MIN_CATCHMENT_KM2,
        ))
}

/// Per-cell mask of which cells emit a river segment for the given set.
fn river_cell_mask(
    hydrology: &hex3::world::Hydrology,
    num_cells: usize,
    set: RiverSet,
) -> Vec<bool> {
    use hex3::world::Hydrology;
    let mode = river_threshold_mode();
    match set {
        RiverSet::All => {
            let flow_threshold = match mode {
                RiverThresholdMode::Legacy => {
                    let (min_flow, _, _) = river_thresholds(num_cells);
                    min_flow * hydrology.mean_cell_discharge
                }
                // Floor at ~4 mean cells so a mesh coarser than the physical
                // threshold (e.g. the 100k coarse mesh, ~5100 km²/cell) doesn't
                // turn every land cell into a river. No effect on the fine mesh
                // (4 cells ≈ 800 km² < any sane min). Naive mean area, NOT
                // mean_cell_discharge — the precip-weighted mean is exactly the
                // ocean-inflated quantity this mode exists to avoid.
                RiverThresholdMode::CatchmentKm2(min) => {
                    let mean_cell_km2 = 4.0
                        * std::f32::consts::PI
                        * hex3::world::diagnostics::EARTH_RADIUS_KM.powi(2)
                        / num_cells.max(1) as f32;
                    Hydrology::flow_for_catchment_km2(min.max(4.0 * mean_cell_km2))
                }
            };
            (0..num_cells)
                .map(|i| {
                    hydrology.flow_accumulation[i] >= flow_threshold && !hydrology.is_submerged(i)
                })
                .collect()
        }
        RiverSet::Major => {
            // `compute_major_river_cells` takes count-equivalents (it scales by
            // mean_cell_discharge internally), so convert physical -> count.
            let (outlet_threshold, branch_threshold) = match mode {
                RiverThresholdMode::Legacy => {
                    let (_, o, b) = river_thresholds(num_cells);
                    (o, b)
                }
                RiverThresholdMode::CatchmentKm2(min) => {
                    let per_count = hydrology.mean_cell_discharge.max(1e-12);
                    (
                        Hydrology::flow_for_catchment_km2(75.0 * min) / per_count,
                        Hydrology::flow_for_catchment_km2(12.5 * min) / per_count,
                    )
                }
            };
            hydrology.compute_major_river_cells(outlet_threshold, branch_threshold)
        }
    }
}

fn river_thresholds(num_cells: usize) -> (f32, f32, f32) {
    let min_flow = (num_cells as f32 * RIVER_MIN_FLOW_FRACTION).max(1.0);
    let outlet_threshold = (num_cells as f32 * RIVER_OUTLET_FRACTION).max(1.0);
    let branch_threshold = (num_cells as f32 * RIVER_BRANCH_FRACTION).max(1.0);
    (min_flow, outlet_threshold, branch_threshold)
}

fn prepare_river_render_data(world: &World) -> Option<RiverRenderData> {
    let hydrology = world.active_hydrology()?;
    let num_cells = world.active_tessellation().num_cells();
    let include_all = river_cell_mask(hydrology, num_cells, RiverSet::All);
    let include_major = river_cell_mask(hydrology, num_cells, RiverSet::Major);
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let max_flow_count = (0..num_cells)
        .map(|i| hydrology.flow_count_equiv(i))
        .fold(0.0f32, f32::max);
    let lake_outflow_paths = hydrology.lake_outflow_paths();

    Some(RiverRenderData {
        include_all,
        include_major,
        max_flow,
        log_max_flow_count: max_flow_count.ln(),
        lake_outflow_paths,
    })
}

fn mode_uses_fine_mesh(world: &World, mode: RenderMode) -> bool {
    world.shows_fine()
        && matches!(
            mode,
            RenderMode::Relief
                | RenderMode::Terrain
                | RenderMode::Elevation
                | RenderMode::Hydrology
                | RenderMode::Climate
        )
}

/// Generate a colored mesh for a specific render mode and layer settings.
/// This is fast (~5-10ms for 80k cells) and called on mode/layer switch.
pub fn generate_colored_mesh(
    device: &wgpu::Device,
    world: &World,
    mode: RenderMode,
    noise_layer: NoiseLayer,
    feature_layer: FeatureLayer,
    climate_layer: ClimateLayer,
) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let use_fine = mode_uses_fine_mesh(world, mode);
    let voronoi = if use_fine {
        &world.active_tessellation().voronoi
    } else {
        &world.tessellation.voronoi
    };

    let mesh = match mode {
        RenderMode::Relief | RenderMode::Terrain => {
            if use_fine {
                VoronoiMesh::from_voronoi_shared_vertices(voronoi, |i| cell_color_terrain(world, i))
            } else {
                VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_terrain(world, i))
            }
        }
        RenderMode::Elevation => {
            if use_fine {
                VoronoiMesh::from_voronoi_shared_vertices(voronoi, |i| {
                    cell_color_elevation(world, i)
                })
            } else {
                VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_elevation(world, i))
            }
        }
        RenderMode::Plates => {
            VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_plate(world, i))
        }
        RenderMode::Noise => VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
            cell_color_noise(world, i, noise_layer)
        }),
        RenderMode::Hydrology => {
            if use_fine {
                VoronoiMesh::from_voronoi_shared_vertices(voronoi, |i| {
                    cell_color_hydrology(world, i)
                })
            } else {
                VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_hydrology(world, i))
            }
        }
        RenderMode::Features => VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
            cell_color_feature(world, i, feature_layer)
        }),
        RenderMode::Climate => {
            if use_fine {
                VoronoiMesh::from_voronoi_shared_vertices(voronoi, |i| {
                    cell_color_climate(world, i, climate_layer)
                })
            } else {
                VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
                    cell_color_climate(world, i, climate_layer)
                })
            }
        }
    };

    let vertex_buffer = create_vertex_buffer(device, &mesh.vertices, "colored_vertex");
    let index_buffer = create_index_buffer(device, &mesh.indices, "colored_index");
    let num_indices = mesh.indices.len() as u32;
    if use_fine {
        let vertex_bytes = buffer_bytes(&mesh.vertices);
        let index_bytes = buffer_bytes(&mesh.indices);
        log::info!(
            "fine mesh GPU colored mesh ({}): vertices={}, indices={}, vertex_bytes={:.1} MiB, index_bytes={:.1} MiB, total={:.1} MiB",
            mode.name(),
            mesh.vertices.len(),
            mesh.indices.len(),
            vertex_bytes as f64 / 1_048_576.0,
            index_bytes as f64 / 1_048_576.0,
            (vertex_bytes + index_bytes) as f64 / 1_048_576.0,
        );
    }

    (vertex_buffer, index_buffer, num_indices)
}

/// Generate GPU buffers from a World.
/// Creates one dynamic colored mesh (initially Terrain mode) plus specialized buffers.
pub fn generate_world_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    world: &World,
) -> WorldBuffers {
    let use_fine = world.shows_fine();
    let voronoi = &world.active_tessellation().voronoi;
    let elevation = world.active_elevation().unwrap();

    let _t = Timed::debug("Build world buffers");

    // Initial colored mesh (Terrain mode - will be regenerated on mode switch)
    let (colored_vertex_buffer, colored_index_buffer, num_colored_indices) = generate_colored_mesh(
        device,
        world,
        RenderMode::Terrain,
        NoiseLayer::Combined,
        FeatureLayer::Trench,
        ClimateLayer::Temperature,
    );

    // Unified mesh with material-aware lighting for Relief mode
    let elevation_for_cell = |i| {
        if let Some(hydrology) = world.active_hydrology() {
            if hydrology.is_ocean(i) {
                return 0.0;
            }
            if hydrology.is_lake_water(i) {
                return hydrology.basin(i).map(|b| b.water_level).unwrap_or(0.0);
            }
        }
        elevation.values[i].max(0.0)
    };

    let unified_mesh = if use_fine {
        UnifiedMesh::from_voronoi_shared_vertices(
            voronoi,
            |i| cell_color_terrain(world, i),
            |i| cell_material(world, i),
            elevation_for_cell,
        )
    } else {
        UnifiedMesh::from_voronoi_with_elevation(
            voronoi,
            |i| cell_color_terrain(world, i),
            |i| cell_material(world, i),
            elevation_for_cell,
        )
    };

    // Edge lines: default gray + plates with colored boundaries
    let edge_color = Vec3::new(0.35, 0.35, 0.35);
    let edge_vertices_default = if use_fine {
        Vec::new()
    } else {
        VoronoiMesh::edge_lines_with_colors(voronoi, |_, _| edge_color)
    };

    let boundary_edge_colors = build_boundary_edge_colors(world);
    let edge_vertices_plates =
        VoronoiMesh::edge_lines_with_colors(&world.tessellation.voronoi, |a, b| {
            let key = if a < b { (a, b) } else { (b, a) };
            boundary_edge_colors
                .get(&key)
                .copied()
                .unwrap_or(edge_color)
        });

    // The relief wireframe is intentionally NOT built here: it is built lazily
    // (see generate_relief_edge_buffers) so its large allocation never overlaps
    // this fill-mesh rebuild, which together would exhaust memory at fine-mesh
    // densities.

    // Plate overlays
    let arrows = generate_velocity_arrows(world);
    let pole_markers = generate_pole_markers(world);

    let arrow_vertices: Vec<MeshVertex> = arrows
        .iter()
        .flat_map(|&(start, end, color)| {
            [
                MeshVertex::new(start, start, color),
                MeshVertex::new(end, end, color),
            ]
        })
        .collect();

    let pole_marker_vertices: Vec<MeshVertex> = pole_markers
        .iter()
        .map(|&(pos, normal, color)| MeshVertex::new(pos, normal, color))
        .collect();
    let pole_marker_indices: Vec<u32> = (0..pole_marker_vertices.len() as u32).collect();

    log::debug!(
        "Overlays: {} arrows, {} pole markers, {} boundary edges",
        arrows.len() / 3,
        pole_markers.len() / 3,
        boundary_edge_colors.len()
    );

    // Rivers
    let river_render_data = prepare_river_render_data(world);
    let river_all_vertices =
        generate_river_vertices(world, river_render_data.as_ref(), RiverSet::All);
    let river_major_vertices =
        generate_river_vertices(world, river_render_data.as_ref(), RiverSet::Major);

    if !river_all_vertices.is_empty() {
        log::debug!(
            "River line segments: {} (major: {})",
            river_all_vertices.len() / 2,
            river_major_vertices.len() / 2,
        );
    }

    if use_fine {
        let unified_vertex_bytes = buffer_bytes(&unified_mesh.vertices);
        let unified_index_bytes = buffer_bytes(&unified_mesh.indices);
        log::info!(
            "fine mesh GPU relief mesh: vertices={}, indices={}, vertex_bytes={:.1} MiB, index_bytes={:.1} MiB, total={:.1} MiB",
            unified_mesh.vertices.len(),
            unified_mesh.indices.len(),
            unified_vertex_bytes as f64 / 1_048_576.0,
            unified_index_bytes as f64 / 1_048_576.0,
            (unified_vertex_bytes + unified_index_bytes) as f64 / 1_048_576.0,
        );
    }

    drop(_t);

    // Draped-river texture: bake the network into an equirect RGBA texture + bind group
    // (group 1 of the unified pipeline). Always built (transparent when no rivers exist).
    let (river_tex_w, river_tex_h) = (8192u32, 4096u32);
    let river_rgba = bake_river_texture(world, river_tex_w, river_tex_h);
    let river_texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("river_texture"),
            size: wgpu::Extent3d {
                width: river_tex_w,
                height: river_tex_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &river_rgba,
    );
    let river_view = river_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let river_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("river_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat, // longitude wraps
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let river_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("river_bind_group"),
        layout: &hex3::render::create_river_bind_group_layout(device),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&river_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&river_sampler),
            },
        ],
    });

    WorldBuffers {
        // Dynamic colored mesh
        colored_vertex_buffer,
        colored_index_buffer,
        num_colored_indices,

        // Edges
        edge_vertex_buffer: create_vertex_buffer(device, &edge_vertices_default, "edge_vertex"),
        edge_vertex_buffer_plates: create_vertex_buffer(
            device,
            &edge_vertices_plates,
            "edge_vertex_plates",
        ),
        num_edge_vertices: edge_vertices_default.len() as u32,
        num_edge_vertices_plates: edge_vertices_plates.len() as u32,

        // Relief mode
        unified_vertex_buffer: create_vertex_buffer(
            device,
            &unified_mesh.vertices,
            "unified_vertex",
        ),
        unified_index_buffer: create_index_buffer(device, &unified_mesh.indices, "unified_index"),
        num_unified_indices: unified_mesh.indices.len() as u32,
        // Built lazily on first show; see generate_relief_edge_buffers.
        relief_edge: None,

        // Rivers
        river_all_vertex_buffer: create_vertex_buffer(
            device,
            &river_all_vertices,
            "river_all_vertex",
        ),
        river_major_vertex_buffer: create_vertex_buffer(
            device,
            &river_major_vertices,
            "river_major_vertex",
        ),
        num_river_all_vertices: river_all_vertices.len() as u32,
        num_river_major_vertices: river_major_vertices.len() as u32,

        river_bind_group,
        _river_texture: river_texture,

        // Plate overlays
        arrow_vertex_buffer: create_vertex_buffer(device, &arrow_vertices, "arrow_vertex"),
        pole_marker_vertex_buffer: create_vertex_buffer(
            device,
            &pole_marker_vertices,
            "pole_marker_vertex",
        ),
        pole_marker_index_buffer: create_index_buffer(
            device,
            &pole_marker_indices,
            "pole_marker_index",
        ),
        num_arrow_vertices: arrow_vertices.len() as u32,
        num_pole_marker_indices: pole_marker_indices.len() as u32,
    }
}

/// Build the relief-mode wireframe (indexed Voronoi edges, elevation displaced).
///
/// Split out of [`generate_world_buffers`] and built lazily (only when edges are
/// shown) so its allocation — hundreds of MiB at the 2.5M-cell fine mesh — never
/// overlaps the fill-mesh rebuild. Building both at once exhausts memory; the
/// crash repro was "advance to the fine mesh with edges already on", whereas
/// "advance, then toggle edges" worked because the wireframe was built alone.
pub fn generate_relief_edge_buffers(device: &wgpu::Device, world: &World) -> ReliefEdgeBuffers {
    let voronoi = &world.active_tessellation().voronoi;
    let elevation = world.active_elevation().unwrap();
    let edge_color = Vec3::new(0.35, 0.35, 0.35);

    let elevation_for_cell = |i| {
        if let Some(hydrology) = world.active_hydrology() {
            if hydrology.is_ocean(i) {
                return 0.0;
            }
            if hydrology.is_lake_water(i) {
                return hydrology.basin(i).map(|b| b.water_level).unwrap_or(0.0);
            }
        }
        elevation.values[i].max(0.0)
    };

    let (vertices, indices) = VoronoiMesh::edge_lines_indexed_with_elevation(
        voronoi,
        edge_color,
        elevation_for_cell,
        |i| cell_material(world, i),
    );

    if world.fine.is_some() {
        log::info!(
            "fine mesh relief wireframe: vertices={}, edges={}, vertex_bytes={:.1} MiB, index_bytes={:.1} MiB",
            vertices.len(),
            indices.len() / 2,
            buffer_bytes(&vertices) as f64 / 1_048_576.0,
            buffer_bytes(&indices) as f64 / 1_048_576.0,
        );
    }

    ReliefEdgeBuffers {
        vertex_buffer: create_vertex_buffer(device, &vertices, "relief_edge_vertex"),
        index_buffer: create_index_buffer(device, &indices, "relief_edge_index"),
        num_indices: indices.len() as u32,
    }
}

/// Generate elevation mesh for rendering to elevation map texture.
/// Returns (vertex_buffer, index_buffer, num_indices).
///
/// Uses the same coastal handling as the unified mesh:
/// - Water cells are flat at their water level
/// - Land cells at water boundary use water level (smooth coast transition)
/// - Interior land uses averaged elevation from adjacent cells
pub fn generate_elevation_mesh_buffers(
    device: &wgpu::Device,
    world: &World,
) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let tessellation = world.active_tessellation();
    let voronoi = &tessellation.voronoi;
    let elevation = world.active_elevation().unwrap();

    // Step 1: Compute per-cell elevation and water status
    let cell_elevations: Vec<f32> = (0..voronoi.num_cells())
        .map(|cell_idx| {
            if let Some(hydrology) = world.active_hydrology() {
                if hydrology.is_ocean(cell_idx) {
                    0.0
                } else if hydrology.is_lake_water(cell_idx) {
                    hydrology
                        .basin(cell_idx)
                        .map(|b| b.water_level)
                        .unwrap_or(0.0)
                } else {
                    elevation.values[cell_idx].max(0.0)
                }
            } else {
                elevation.values[cell_idx].max(0.0)
            }
        })
        .collect();

    let cell_is_water: Vec<bool> = (0..voronoi.num_cells())
        .map(|cell_idx| {
            if let Some(hydrology) = world.active_hydrology() {
                hydrology.is_ocean(cell_idx) || hydrology.is_lake_water(cell_idx)
            } else {
                elevation.values[cell_idx] <= 0.0
            }
        })
        .collect();

    // Step 2: For each vertex, track water level and land elevation statistics
    let mut vertex_land_sum = vec![0.0f32; voronoi.vertices.len()];
    let mut vertex_land_count = vec![0u32; voronoi.vertices.len()];
    let mut vertex_water_level = vec![None::<f32>; voronoi.vertices.len()];

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let elev = cell_elevations[cell_idx];
        let is_water = cell_is_water[cell_idx];

        for &vertex_idx in cell.vertex_indices {
            let vi = vertex_idx as usize;
            if is_water {
                // Track water level (use max in case of adjacent lakes at different levels)
                vertex_water_level[vi] = Some(
                    vertex_water_level[vi]
                        .map(|wl| wl.max(elev))
                        .unwrap_or(elev),
                );
            } else {
                // Accumulate land elevation for averaging
                vertex_land_sum[vi] += elev;
                vertex_land_count[vi] += 1;
            }
        }
    }

    // Step 3: Compute final per-vertex elevations with water boundary handling
    let vertex_elevations: Vec<f32> = (0..voronoi.vertices.len())
        .map(|v| {
            let water_level = vertex_water_level[v];
            let land_count = vertex_land_count[v];

            match (water_level, land_count) {
                // All water: use water level
                (Some(wl), 0) => wl,
                // All land: average land elevations
                (None, n) if n > 0 => vertex_land_sum[v] / n as f32,
                // Mixed (land touching water): use max of land average and water level
                (Some(wl), n) if n > 0 => {
                    let land_avg = vertex_land_sum[v] / n as f32;
                    land_avg.max(wl)
                }
                _ => 0.0,
            }
        })
        .collect();

    // Step 4: Build a shared-vertex mesh with proper coastal elevation handling.
    let mut vertices = Vec::with_capacity(voronoi.vertices.len());
    for (vi, &pos) in voronoi.vertices.iter().enumerate() {
        vertices.push(ElevationVertex {
            position: [pos.x, pos.y, pos.z],
            elevation: vertex_elevations[vi],
        });
    }
    let mut indices = Vec::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        if cell.len() < 3 {
            continue;
        }

        // Fan triangulation
        let n = cell.vertex_indices.len();
        for i in 1..n - 1 {
            indices.push(cell.vertex_indices[0]);
            indices.push(cell.vertex_indices[i]);
            indices.push(cell.vertex_indices[i + 1]);
        }
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elevation_mesh_vertex_buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elevation_mesh_index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    (vertex_buffer, index_buffer, indices.len() as u32)
}

/// Generate line-based river vertices (used outside Relief mode).
/// Alpha encodes flow magnitude on a log scale.
fn generate_river_vertices(
    world: &World,
    render_data: Option<&RiverRenderData>,
    set: RiverSet,
) -> Vec<SurfaceVertex> {
    let Some(hydrology) = world.active_hydrology() else {
        return Vec::new();
    };
    let Some(render_data) = render_data else {
        return Vec::new();
    };
    let elevation = world.active_elevation().unwrap();
    let tessellation = world.active_tessellation();

    let include = render_data.include(set);
    let log_max = render_data.log_max_flow_count;

    let mut vertices = Vec::with_capacity(include.iter().filter(|&&included| included).count() * 2);

    for (cell_idx, &included) in include.iter().enumerate() {
        if !included {
            continue;
        }

        let Some(downstream_idx) = hydrology.downstream(cell_idx) else {
            continue;
        };

        let (start_pos, end_pos, start_elev, end_elev) =
            river_segment_geometry(tessellation, elevation, hydrology, cell_idx, downstream_idx);

        // Alpha based on logarithmic flow
        let flow = hydrology.flow_count_equiv(cell_idx);
        let alpha = 0.15 + 0.55 * (flow.ln() / log_max).clamp(0.0, 1.0);

        vertices.push(SurfaceVertex::new(
            start_pos,
            start_elev,
            RIVER_COLOR,
            alpha,
        ));
        vertices.push(SurfaceVertex::new(end_pos, end_elev, RIVER_COLOR, alpha));
    }

    // Add lake outflow rivers (from overflowing lakes)
    generate_lake_outflow_vertices(
        world,
        &render_data.lake_outflow_paths,
        &mut vertices,
        RIVER_COLOR,
    );

    vertices
}

/// Helper to compute geometry for a river segment (line-based rendering).
fn river_segment_geometry(
    tessellation: &hex3::world::Tessellation,
    elevation: &hex3::world::Elevation,
    hydrology: &hex3::world::Hydrology,
    cell_idx: usize,
    downstream_idx: usize,
) -> (Vec3, Vec3, f32, f32) {
    let start_center = tessellation.cell_center(cell_idx);
    let end_center = tessellation.cell_center(downstream_idx);

    let start_elev = elevation.values[cell_idx];
    let end_elev = if hydrology.is_submerged(downstream_idx) {
        if hydrology.is_ocean(downstream_idx) {
            0.0
        } else {
            hydrology
                .basin(downstream_idx)
                .map(|b| b.water_level)
                .unwrap_or(0.0)
        }
    } else {
        elevation.values[downstream_idx]
    };

    let end_pos = if hydrology.is_submerged(downstream_idx) {
        ((start_center + end_center) / 2.0).normalize()
    } else {
        end_center
    };

    (start_center, end_pos, start_elev, end_elev)
}

/// Generate vertices for lake outflow rivers (from overflowing lakes).
/// These are added to the existing vertices vector.
fn generate_lake_outflow_vertices(
    world: &World,
    lake_outflow_paths: &[(usize, Vec<usize>)],
    vertices: &mut Vec<SurfaceVertex>,
    river_color: Vec3,
) {
    let Some(hydrology) = world.active_hydrology() else {
        return;
    };
    let elevation = world.active_elevation().unwrap();
    let tessellation = world.active_tessellation();

    // Lake outflows are significant rivers - use high alpha
    let outflow_alpha = 0.7;

    for (_basin_idx, path) in lake_outflow_paths {
        // Generate segments along the outflow path
        for window in path.windows(2) {
            let cell_idx = window[0];
            let downstream_idx = window[1];

            let (start_pos, end_pos, start_elev, end_elev) = river_segment_geometry(
                tessellation,
                elevation,
                hydrology,
                cell_idx,
                downstream_idx,
            );

            vertices.push(SurfaceVertex::new(
                start_pos,
                start_elev,
                river_color,
                outflow_alpha,
            ));
            vertices.push(SurfaceVertex::new(
                end_pos,
                end_elev,
                river_color,
                outflow_alpha,
            ));
        }

        // Add final segment to water (if path ends at land cell adjacent to water)
        if let Some(&last_cell) = path.last() {
            if let Some(downstream_idx) = hydrology.downstream(last_cell) {
                if hydrology.is_submerged(downstream_idx) {
                    let (start_pos, end_pos, start_elev, end_elev) = river_segment_geometry(
                        tessellation,
                        elevation,
                        hydrology,
                        last_cell,
                        downstream_idx,
                    );

                    vertices.push(SurfaceVertex::new(
                        start_pos,
                        start_elev,
                        river_color,
                        outflow_alpha,
                    ));
                    vertices.push(SurfaceVertex::new(
                        end_pos,
                        end_elev,
                        river_color,
                        outflow_alpha,
                    ));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// River TEXTURE bake (river-render re-work): rasterize the river network into an
// equirectangular RGBA mask so the terrain shader can draw rivers AS SURFACE SHADING
// (perfectly draped) instead of floating quad ribbons. This reuses the same network +
// flow-widths the quads used; only the rendering changes.
// ---------------------------------------------------------------------------

/// World position (unit sphere) → equirect pixel (u,v). Convention must match the shader.
fn river_uv(p: Vec3, w: usize, h: usize) -> (f32, f32) {
    let p = p.normalize();
    let lat = p.y.clamp(-1.0, 1.0).asin(); // [-π/2, π/2]
    let lon = p.z.atan2(p.x); // [-π, π]
    let u = (lon / (2.0 * std::f32::consts::PI) + 0.5) * w as f32;
    let v = (0.5 - lat / std::f32::consts::PI) * h as f32;
    (u, v)
}

/// Distance range (pixels) the SDF encodes: R=0 on the river centerline → R=255 at
/// `RIVER_SDF_RANGE_PX` or beyond. MUST match `RIVER_SDF_RANGE_PX` in unified.wgsl.
const RIVER_SDF_RANGE_PX: f32 = 6.0;

/// Stamp the unsigned distance-to-point into R (keeping the minimum), the nearest river's
/// flow into G, and whether that nearest river is a MAJOR river into B (so the shader can
/// honour the Off/Major/All density mode). Longitude wraps at the seam.
fn stamp_distance(buf: &mut [u8], w: usize, h: usize, cu: f32, cv: f32, flow_u8: u8, major: bool) {
    let r1 = RIVER_SDF_RANGE_PX + 1.0;
    let major_u8 = if major { 255 } else { 0 };
    for vv in (cv - r1).floor() as i32..=(cv + r1).ceil() as i32 {
        if vv < 0 || vv >= h as i32 {
            continue;
        }
        for uu in (cu - r1).floor() as i32..=(cu + r1).ceil() as i32 {
            let d = (((uu as f32) - cu).powi(2) + ((vv as f32) - cv).powi(2)).sqrt();
            if d > RIVER_SDF_RANGE_PX {
                continue;
            }
            let d_u8 = (d / RIVER_SDF_RANGE_PX * 255.0) as u8;
            let px = uu.rem_euclid(w as i32) as usize;
            let idx = (vv as usize * w + px) * 4;
            if d_u8 < buf[idx] {
                buf[idx] = d_u8; // R = distance to centerline (0 = on river)
                buf[idx + 1] = flow_u8; // G = nearest river's flow factor
                buf[idx + 2] = major_u8; // B = nearest river is a major river
            }
        }
    }
}

/// Rasterize one river segment a→b (unit-sphere endpoints) into the distance field,
/// walking the shortest path across the longitude seam.
fn stamp_segment(buf: &mut [u8], w: usize, h: usize, a: Vec3, b: Vec3, flow_u8: u8, major: bool) {
    let (u0, v0) = river_uv(a, w, h);
    let (mut u1, v1) = river_uv(b, w, h);
    if (u1 - u0).abs() > w as f32 * 0.5 {
        if u1 > u0 {
            u1 -= w as f32;
        } else {
            u1 += w as f32;
        }
    }
    let len = (u1 - u0).hypot(v1 - v0).max(1.0);
    let steps = len.ceil() as usize;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        stamp_distance(
            buf,
            w,
            h,
            u0 + (u1 - u0) * t,
            v0 + (v1 - v0) * t,
            flow_u8,
            major,
        );
    }
}

/// Bake the river network into an equirectangular RGBA SDF: R = distance-to-river (0 = on a
/// river, 255 = far), G = nearest river's flow factor (downstream widening), B = nearest
/// river is MAJOR (the Off/Major/All density mode is a shader toggle on this, not a re-bake).
/// The terrain shader reconstructs thin, crisp, screen-space-AA'd rivers from this.
pub fn bake_river_texture(world: &World, width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    // R = 255 (far / no river) everywhere; G/B/A = 0.
    let mut buf = vec![0u8; w * h * 4];
    for texel in buf.chunks_mut(4) {
        texel[0] = 255;
    }
    let (Some(render_data), Some(hydrology)) =
        (prepare_river_render_data(world), world.active_hydrology())
    else {
        return buf;
    };
    let tess = world.active_tessellation();
    let max_flow = render_data.max_flow.max(1e-6);
    let include_all = render_data.include(RiverSet::All);
    let include_major = render_data.include(RiverSet::Major);

    for (i, &on) in include_all.iter().enumerate() {
        if !on {
            continue;
        }
        let Some(j) = hydrology.downstream(i) else {
            continue;
        };
        let start = tess.cell_center(i);
        let end = if hydrology.is_submerged(j) {
            ((start + tess.cell_center(j)) / 2.0).normalize()
        } else {
            tess.cell_center(j)
        };
        let flow_u8 = ((hydrology.flow_accumulation[i] / max_flow).sqrt() * 255.0) as u8;
        stamp_segment(&mut buf, w, h, start, end, flow_u8, include_major[i]);
    }

    // Lake outflow channels — treat as major rivers (they carry a basin's full discharge).
    for (_basin, path) in &render_data.lake_outflow_paths {
        for seg in path.windows(2) {
            stamp_segment(
                &mut buf,
                w,
                h,
                tess.cell_center(seg[0]),
                tess.cell_center(seg[1]),
                200,
                true,
            );
        }
    }
    buf
}

fn print_world_stats(world: &World) {
    let elevation = world.elevation.as_ref().unwrap();
    let num_cells = world.num_cells();

    let water_count = elevation.values.iter().filter(|&&e| e < 0.0).count();
    let water_pct = 100.0 * water_count as f32 / num_cells as f32;
    let avg_elevation: f32 = elevation.values.iter().sum::<f32>() / num_cells as f32;
    let min_elevation = elevation
        .values
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_elevation = elevation
        .values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let continental_pct = 100.0
        * world
            .crust
            .as_ref()
            .map(|c| c.continental_fraction())
            .unwrap_or(0.0);

    log::info!(
        "Stats: cells={}, water={:.1}%, continental={:.1}%, elev=[{:.3}, {:.3}], avg={:.3}",
        num_cells,
        water_pct,
        continental_pct,
        min_elevation,
        max_elevation,
        avg_elevation
    );
}
