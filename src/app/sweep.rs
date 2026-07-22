//! Headless parameter-sweep harness: render the stage-4 relief view to PNG tiles
//! for a grid of erosion-knob values, plus a stitched contour-sheet montage, so
//! knob effects can be eyeballed without driving the interactive app frame by
//! frame. Runs on the GPU but needs no window (offscreen render-to-texture +
//! readback), so a single `cargo run` produces the whole image grid.
//!
//! The fine-mesh base (stage 3a) is a function of seed + mesh density + the
//! structural-relief knobs (P1a), so it is disk-cached and reused across every
//! tile; for a pure EROSION-knob sweep only the erosion stage re-runs per value,
//! which is what makes a sweep affordable after the perf work. Sweeping a
//! structural knob (`fault_scarp`, `interior_relief`) regenerates the base per
//! value by design (decision A) — distinct key per value, so still cache-correct,
//! just not free.

use std::io::BufWriter;
use std::path::PathBuf;
use std::str::FromStr;
use std::{collections::BTreeSet, time::Instant};

use glam::{Mat4, Vec3};
use serde::Serialize;
use wgpu::util::DeviceExt;

use hex3::render::{
    create_index_buffer, create_vertex_buffer, FillPipelineKind, GpuContext, IndexedDraw,
    OrbitCamera, RenderScene, Renderer, SurfaceLineDraw, Uniforms,
};
use hex3::{
    geometry::{Material, SurfaceVertex, UnifiedMesh, VoronoiMesh},
    world::{
        assess_route_lower_corridor, build_aggregate_route_network, AggregateRouteNetwork,
        AggregateSiteSelection, BasinSpillDestination, ConsequentialGeographyComponents,
        FineCacheMode, FreshwaterSourceKind, LivingSurfaceSemantics, OrogenModel, RiverNetwork,
        RiverSelection, RiverThresholdPolicy, RouteLowerCorridorAssessment,
        RouteLowerCorridorEvidence, RouteLowerCorridorOmission, RouteNetworkConfig,
        SemanticWaterKind, ShorelineLoop, SiteSelectionConfig, Tessellation, TraversalConfig,
        VoronoiBackend, WaterBodyId, WaterBodySemantics, WaterGeographyGeometry, World,
        PLANET_RADIUS_KM, RELIEF_SCALE,
    },
};

#[cfg(feature = "research-landscape")]
use hex3::world::{
    build_regional_deformation_overlap_map_v0, build_regional_deformation_rds0_v0,
    collect_convergent_fronts, collect_plate_boundaries, evaluate_regional_deformation_frame_v0,
    evaluate_regional_deformation_static_control_v0,
    transfer_regional_deformation_raster_with_overlap_v0, B0DrainageDualResultV0, FineSurface,
    LegacyBudgetOpportunityAuditV0, RegionalDeformationProgramV0,
    RegionalDeformationRasterLedgerV0, RegionalDeformationRasterV0, PHYSICAL_RELIEF_SCALE,
    RDS0_FRAME_COUNT,
};

use super::coloring::{
    cell_color_terrain, cell_material, living_surface_blended_color, LIVING_HERBACEOUS_COLOR,
    LIVING_WETLAND_COLOR, LIVING_WOODY_COLOR,
};
#[cfg(feature = "research-landscape")]
use super::rds_relationship::{
    analyze_rds0_relationships_v0, compress_rds0_schedule_v0, RdsRelationshipAnalysisV0,
};
use super::view::{ReliefPreset, RiverMode, SurfacePalette};
#[cfg(feature = "research-landscape")]
use super::world::regenerate_diagnostic_surface_colors;
use super::world::{
    advance_to_stage_2, advance_to_stage_3, advance_to_stage_3_with_cap, advance_to_stage_4,
    create_world_with_orogen_model, generate_world_buffers,
    generate_world_buffers_with_display_subdivision, regenerate_surface_palette, ErosionOverrides,
};

/// Knobs the sweep can vary, mapped onto [`ErosionOverrides`] fields.
pub const SWEEP_KNOBS: &[&str] = &[
    "relief_scale",
    "river_width_scale",
    "k",
    "n",
    "diffusivity",
    "channel_support",
    "hillslope_crit",
    "confinement_slope",
    "uplift_smooth",
    "mfd_exponent",
    "diffusion_iters",
    "reroute_interval",
    "steps",
    "precip_iters",
    "flat_resolution",
    "uplift_scale",
    "deposition_slope",
    "litho_sigma",
    "litho_grain",
    "orographic",
    "downwind_shadow",
    "lake_evap",
    "climate",
    "glacial_k",
    "fault_scarp",
    "interior_relief",
    "front_strike_weight",
    "margin_contrast",
    "emergent_lambda",
    "emergent_structured",
    "rebuild_gain",
    "meso_relief",
    "meso_irregularity",
    "meso_style",
    "meso_base_relief",
    "meso_wavelength_km",
    "drainage_pulse",
    "pulse_burnin_steps",
    "pulse_smooth_km",
    "orogen_model",
];

/// A stable, human-named camera target supplied as `id:lat_deg:lon_deg`.
///
/// Hex3 uses Y as the polar axis and longitude `atan2(z, x)`: longitude zero is
/// +X and positive longitude rotates toward +Z.
#[derive(Clone, Debug, Serialize)]
pub struct SweepTarget {
    pub id: String,
    pub latitude_deg: f32,
    pub longitude_deg: f32,
}

impl SweepTarget {
    fn unit_position(&self) -> Vec3 {
        let lat = self.latitude_deg.to_radians();
        let lon = self.longitude_deg.to_radians();
        Vec3::new(lat.cos() * lon.cos(), lat.sin(), lat.cos() * lon.sin())
    }
}

impl FromStr for SweepTarget {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut fields = value.split(':');
        let id = fields.next().unwrap_or_default();
        let latitude = fields.next();
        let longitude = fields.next();
        if id.is_empty() || latitude.is_none() || longitude.is_none() || fields.next().is_some() {
            return Err("expected id:lat_deg:lon_deg".to_string());
        }
        if !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        {
            return Err(
                "target id may contain only ASCII letters, digits, '-' and '_'".to_string(),
            );
        }
        let latitude_deg = latitude
            .unwrap()
            .parse::<f32>()
            .map_err(|_| "latitude must be a number in degrees".to_string())?;
        let longitude_deg = longitude
            .unwrap()
            .parse::<f32>()
            .map_err(|_| "longitude must be a number in degrees".to_string())?;
        if !latitude_deg.is_finite() || !(-90.0..=90.0).contains(&latitude_deg) {
            return Err("latitude must be finite and between -90 and 90 degrees".to_string());
        }
        if !longitude_deg.is_finite() || !(-180.0..=180.0).contains(&longitude_deg) {
            return Err("longitude must be finite and between -180 and 180 degrees".to_string());
        }
        Ok(Self {
            id: id.to_string(),
            latitude_deg,
            longitude_deg,
        })
    }
}

/// Options for a sweep run, assembled from the CLI.
pub struct SweepOptions {
    pub seed: u64,
    pub cells: usize,
    pub fine_scale: f32,
    /// Fine-mesh cell guardrail. Zero uses the product/default budget.
    pub fine_max: usize,
    pub target_stage: u32,
    pub voronoi_backend: VoronoiBackend,
    pub orogen_model: OrogenModel,
    pub fine_cache: FineCacheMode,
    /// Baseline overrides applied to every tile (the non-swept knobs).
    pub base_erosion: ErosionOverrides,
    /// Preset cumulative stack (e.g. "p1"): renders a fixed sequence of knob
    /// combinations (each rung layered on the previous) instead of a single-knob
    /// sweep, sharing one camera set so the rungs are directly comparable. When
    /// `Some`, `knob1`/`values1`/`knob2` are ignored.
    pub stack: Option<String>,
    /// Knob varied across columns and its values.
    pub knob1: String,
    pub values1: Vec<f64>,
    /// Optional knob varied across rows (grid); empty values => 1-D sweep.
    pub knob2: Option<String>,
    pub values2: Vec<f64>,
    pub out_dir: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Overview (globe) camera: orbit angle + distance from center.
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub distance: f32,
    /// Number of zoomed close-up views per tile, auto-aimed at the highest land
    /// (in addition to the globe overview). 0 = overview only.
    pub zoom_views: usize,
    /// Close-up camera altitude above the target (smaller = tighter zoom).
    pub zoom_alt: f32,
    /// Stable dossier targets. When non-empty these replace automatically
    /// selected highland close-ups; the globe overview is still included.
    pub targets: Vec<SweepTarget>,
    pub river_mode: RiverMode,
    pub surface_palette: SurfacePalette,
    pub river_threshold_policy: String,
    pub river_min_catchment_km2: Option<f32>,
    /// Research-only render subdivision. It interpolates an unchanged physical
    /// surface and is therefore meaningful only as a globe display A/B.
    pub display_subdivision_levels: usize,
}

/// Apply one knob=value onto the overrides. Errors on an unknown knob name.
fn apply_knob(ov: &mut ErosionOverrides, name: &str, v: f64) -> Result<(), String> {
    let f = v as f32;
    match name {
        "relief_scale" => ov.relief_scale = Some(f.max(0.0)),
        "river_width_scale" => ov.river_width_scale = Some(f.max(0.0)),
        "k" => ov.k = Some(f),
        "n" => ov.n = Some(f),
        "diffusivity" => ov.diffusivity = Some(f),
        "channel_support" => ov.channel_support_km2 = Some(f),
        "hillslope_crit" => ov.hillslope_critical_slope = Some(f),
        "confinement_slope" => ov.confinement_slope = Some(f),
        "uplift_smooth" => ov.uplift_smooth_km = Some(f),
        "mfd_exponent" => ov.mfd_exponent = Some(f),
        "diffusion_iters" => ov.diffusion_iters = Some(v as usize),
        "reroute_interval" => ov.reroute_interval = Some(v as usize),
        "steps" => ov.steps = Some(v as usize),
        "precip_iters" => ov.precip_outer_iters = Some(v as usize),
        "flat_resolution" => ov.flat_resolution = Some(v != 0.0),
        "uplift_scale" => ov.uplift_scale = Some(f),
        "deposition_slope" => ov.deposition_slope = Some(f),
        "litho_sigma" => ov.litho_sigma = Some(f),
        "litho_grain" => ov.litho_grain_strength = Some(f),
        "orographic" => ov.orographic_precip_strength = Some(f),
        "downwind_shadow" => ov.downwind_shadow_strength = Some(f),
        "lake_evap" => ov.lake_evap_strength = Some(f),
        "climate" => ov.climate_ratio = Some(f),
        "glacial_k" => ov.glacial_k = Some(f),
        "fault_scarp" => ov.fault_scarp_height = Some(f),
        "interior_relief" => ov.interior_relief = Some(f),
        "front_strike_weight" => ov.front_strike_weight = Some(f),
        "margin_contrast" => ov.margin_contrast = Some(f),
        "emergent_lambda" => ov.emergent_lambda = Some(f),
        "emergent_structured" => ov.emergent_structured = Some(f),
        "rebuild_gain" => ov.rebuild_gain = Some(f),
        "meso_relief" => ov.meso_relief = Some(f),
        "meso_irregularity" => ov.meso_irregularity = Some(f),
        "meso_style" => ov.meso_style = Some(v as usize),
        "meso_base_relief" => ov.meso_base_relief = Some(f),
        "meso_wavelength_km" => ov.meso_wavelength_km = Some(f),
        "drainage_pulse" => ov.drainage_pulse = Some(f),
        "pulse_burnin_steps" => ov.pulse_burnin_steps = Some(v as usize),
        "pulse_smooth_km" => ov.pulse_smooth_km = Some(f),
        // Categorical: 0 = legacy, 1 = legacy-yield (the pillar A/B). The
        // experimental conserved/thin-sheet rungs are diagnose-only.
        "orogen_model" => {
            ov.orogen_model = Some(match v as usize {
                0 => hex3::world::OrogenModel::Legacy,
                1 => hex3::world::OrogenModel::LegacyYield,
                other => {
                    return Err(format!(
                        "orogen_model sweep value {other} (0=legacy, 1=legacy-yield)"
                    ))
                }
            })
        }
        other => {
            return Err(format!(
                "unknown sweep knob '{other}'; valid knobs: {}",
                SWEEP_KNOBS.join(", ")
            ))
        }
    }
    Ok(())
}

fn selected_orogen_model(default: OrogenModel, overrides: &ErosionOverrides) -> OrogenModel {
    overrides.orogen_model.unwrap_or(default)
}

/// Generate a fully-staged world for one tile's knob values.
fn generate_tile_world(opts: &SweepOptions, overrides: &ErosionOverrides) -> World {
    // Stage 1 owns the orogen model, so resolve a swept categorical override
    // before constructing the world. Applying it only afterward leaves feature
    // and elevation fields from the baseline model under a mismatched manifest.
    let orogen_model = selected_orogen_model(opts.orogen_model, overrides);
    let mut world = create_world_with_orogen_model(
        opts.seed,
        opts.cells,
        opts.voronoi_backend,
        opts.fine_cache,
        orogen_model,
    );
    overrides.apply(&mut world);

    if (opts.fine_scale - 1.0).abs() > f32::EPSILON {
        let dp = &mut world.fine_density_params;
        dp.plains_km *= opts.fine_scale;
        dp.mountain_km *= opts.fine_scale;
        dp.ocean_km *= opts.fine_scale;
        world.fine_cache = FineCacheMode::Disabled;
    }

    if opts.target_stage >= 2 {
        advance_to_stage_2(&mut world);
    }
    if opts.target_stage >= 3 {
        if opts.fine_max > 0 {
            advance_to_stage_3_with_cap(&mut world, opts.fine_max);
        } else {
            advance_to_stage_3(&mut world);
        }
    }
    if opts.target_stage >= 4 {
        advance_to_stage_4(&mut world);
    }
    // Climate-ratio sweep: adjust lake levels on the (already-integrated) drainage post-gen.
    if let Some(c) = overrides.climate_ratio {
        world.set_active_climate_ratio(c);
    }
    world
}

/// Render a world's relief view (from a prebuilt view-projection + eye) into
/// `color_view`. `buffers` is built once per tile and shared across its views.
fn render_relief(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_view: &wgpu::TextureView,
    buffers: &super::world::WorldBuffers,
    view_proj: Mat4,
    cam_pos: Vec3,
    river_mode: RiverMode,
    relief_scale: f32,
    river_width_scale: f32,
) {
    let light = Vec3::new(0.5, 1.0, 0.3).normalize();
    let uniforms = Uniforms::new(view_proj, cam_pos, light)
        .with_relief_scale(relief_scale)
        // Hillshade from the displaced face normal + simple directional light, so
        // terrain SLOPES are legible (the relief-judging view): hemisphere lighting +
        // sphere-normal shading washed peaks out to flat white. See unified.wgsl.
        .with_slope_shading(true)
        .with_hemisphere_lighting(false)
        .with_map_mode(false)
        .with_rivers(river_mode != RiverMode::Off)
        .with_river_major_only(river_mode == RiverMode::Major)
        .with_river_width_scale(river_width_scale);

    let scene = RenderScene {
        fill_pipeline: FillPipelineKind::UnifiedGlobe,
        fill: IndexedDraw {
            vertex_buffer: &buffers.unified_vertex_buffer,
            index_buffer: &buffers.unified_index_buffer,
            index_count: buffers.num_unified_indices,
        },
        river_texture_bind_group: Some(&buffers.river_bind_group),
        edges: None,
        arrows: None,
        pole_markers: None,
        rivers: None,
        gpu_particles: None,
    };

    renderer.render_to_view(&gpu.device, &gpu.queue, color_view, &uniforms, scene);
}

/// Copy the rendered color texture back to CPU as tight (unpadded) RGBA8 bytes.
fn read_back_rgba(gpu: &GpuContext, color_tex: &wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
    let bpp = 4u32;
    let unpadded = width * bpp;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;

    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sweep_readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sweep_readback_encoder"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: color_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(std::iter::once(encoder.finish()));

    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| {
        r.expect("sweep readback map failed");
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("sweep readback poll failed");

    let data = buffer.slice(..).get_mapped_range();
    let mut tight = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        tight.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    tight
}

/// Write tight RGBA8 bytes to a PNG.
fn write_png(path: &std::path::Path, rgba: &[u8], width: u32, height: u32) {
    let file =
        std::fs::File::create(path).unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer.write_image_data(rgba).expect("png data");
}

/// Filename-safe rendering of a knob value (e.g. 0.04 -> "0.04", 30.0 -> "30").
fn fmt_value(v: f64) -> String {
    let s = format!("{v}");
    s.replace(['/', ' '], "_")
}

/// Blit a tile's RGBA into the montage at grid cell (col, row).
fn blit_tile(
    montage: &mut [u8],
    montage_w: u32,
    tile: &[u8],
    tile_w: u32,
    tile_h: u32,
    col: u32,
    row: u32,
) {
    let bpp = 4usize;
    for y in 0..tile_h {
        let dst_x = col * tile_w;
        let dst_y = row * tile_h + y;
        let dst = ((dst_y * montage_w + dst_x) as usize) * bpp;
        let src = (y * tile_w) as usize * bpp;
        let len = tile_w as usize * bpp;
        montage[dst..dst + len].copy_from_slice(&tile[src..src + len]);
    }
}

/// Pick up to `k` close-up camera targets (unit-sphere positions): the highest
/// land cells, greedily spread apart so the views cover distinct regions rather
/// than clustering on one massif.
fn pick_targets(world: &World, k: usize) -> Vec<Vec3> {
    if k == 0 {
        return Vec::new();
    }
    let tess = world.active_tessellation();
    let Some(elev) = world.active_elevation() else {
        return Vec::new();
    };
    let elev = &elev.values;
    let n = tess.num_cells();
    let mut land: Vec<(usize, f32)> = (0..n)
        .filter(|&i| elev[i] > 0.0)
        .map(|i| (i, elev[i]))
        .collect();
    if land.is_empty() {
        return Vec::new();
    }
    land.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    // High-elevation pool to spread within (the dissected orogens live here).
    let pool: Vec<usize> = land.iter().take(4000).map(|&(i, _)| i).collect();
    let pos = |i: usize| tess.cell_center(i);

    let mut picked = vec![pool[0]];
    while picked.len() < k {
        let mut best = None;
        let mut best_d = -1.0f32;
        for &c in &pool {
            if picked.contains(&c) {
                continue;
            }
            let pc = pos(c);
            let dmin = picked
                .iter()
                .map(|&p| (pos(p) - pc).length())
                .fold(f32::INFINITY, f32::min);
            if dmin > best_d {
                best_d = dmin;
                best = Some(c);
            }
        }
        match best {
            Some(c) => picked.push(c),
            None => break,
        }
    }
    picked.into_iter().map(pos).collect()
}

/// Oblique aerial camera looking down at a surface point (unit-sphere direction),
/// raised `alt` along the surface normal and offset along a tangent so relief
/// reads in profile. Smaller `alt` = tighter zoom.
fn target_camera(center_unit: Vec3, aspect: f32, alt: f32) -> (Mat4, Vec3) {
    let n = center_unit.normalize();
    // Aim a touch above the mean surface (relief peaks sit above radius 1).
    let target = n * 1.08;
    let up_ref = if n.y.abs() < 0.95 { Vec3::Y } else { Vec3::Z };
    let east = n.cross(up_ref).normalize();
    let north = east.cross(n).normalize();
    // Raised along the normal, offset less along the tangent => steeper look-down
    // so the limb/sky stays mostly out of frame and terrain fills it.
    let eye = target + n * alt + north * (alt * 0.55);
    let view = Mat4::look_at_rh(eye, target, n);
    let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 10.0);
    (proj * view, eye)
}

#[derive(Debug)]
struct CaptureView {
    view_proj: Mat4,
    eye: Vec3,
    label: String,
    sidecar: ViewRecord,
}

#[derive(Clone, Debug, Serialize)]
struct ViewRecord {
    id: String,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    target: Option<SweepTarget>,
    camera: CameraRecord,
}

#[derive(Clone, Debug, Serialize)]
struct CameraRecord {
    eye_xyz: [f32; 3],
    aim_xyz: [f32; 3],
    up_xyz: [f32; 3],
    vertical_fov_deg: f32,
    aspect: f32,
    near: f32,
    far: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_altitude: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    orbit_yaw_deg: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    orbit_pitch_deg: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    orbit_distance: Option<f32>,
}

#[derive(Debug, Serialize)]
struct CaptureSidecar {
    schema_version: u32,
    coordinate_convention: &'static str,
    config: CaptureConfig,
    views: Vec<ViewRecord>,
    tiles: Vec<TileRecord>,
    montage_filename: &'static str,
    future_layers: [&'static str; 1],
}

#[derive(Debug, Serialize)]
struct CaptureConfig {
    seed: u64,
    requested_coarse_cells: usize,
    fine_scale: f32,
    fine_max: usize,
    target_stage: u32,
    voronoi_backend: VoronoiBackend,
    orogen_model: OrogenModel,
    fine_cache: FineCacheMode,
    viewport_width: u32,
    viewport_height: u32,
    sweep_stack: Option<String>,
    primary_knob: String,
    primary_values: Vec<f64>,
    secondary_knob: Option<String>,
    secondary_values: Vec<f64>,
    overview_yaw_deg: f32,
    overview_pitch_deg: f32,
    overview_distance: f32,
    closeup_altitude: f32,
    explicit_targets: Vec<SweepTarget>,
    automatic_zoom_views_if_no_targets: usize,
    river_mode: &'static str,
    surface_palette: &'static str,
    river_threshold_policy: String,
    river_min_catchment_km2: Option<f32>,
    display_subdivision_levels: usize,
}

#[derive(Debug, Serialize)]
struct TileRecord {
    index: usize,
    label: String,
    filename_stem: String,
    knob_values: Vec<KnobValue>,
    relief_scale: f32,
    river_width_scale: f32,
    image_filenames: Vec<String>,
    world_manifest: hex3::world::RunManifest,
}

#[derive(Debug, Serialize)]
struct KnobValue {
    name: String,
    value: f64,
}

fn vec3_array(v: Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

fn river_mode_label(mode: RiverMode) -> &'static str {
    match mode {
        RiverMode::Off => "off",
        RiverMode::Major => "major",
        RiverMode::All => "all",
    }
}

/// Build the per-tile view set: a globe overview plus `zoom_views` close-ups
/// aimed at the highest land in `world`. The same set is reused for every tile so
/// each montage column shows the same region/angle across knob values.
fn build_views(world: &World, opts: &SweepOptions) -> Vec<CaptureView> {
    let aspect = opts.width as f32 / opts.height as f32;
    let mut views = Vec::new();

    let mut cam = OrbitCamera::new();
    cam.yaw = opts.yaw_deg.to_radians();
    cam.pitch = opts.pitch_deg.to_radians();
    cam.distance = opts.distance;
    cam.aspect = aspect;
    let eye = cam.eye_position();
    views.push(CaptureView {
        view_proj: cam.view_projection(),
        eye,
        label: "globe".to_string(),
        sidecar: ViewRecord {
            id: "globe".to_string(),
            kind: "overview",
            target: None,
            camera: CameraRecord {
                eye_xyz: vec3_array(eye),
                aim_xyz: [0.0, 0.0, 0.0],
                up_xyz: [0.0, 1.0, 0.0],
                vertical_fov_deg: 45.0,
                aspect,
                near: 0.01,
                far: 10.0,
                target_altitude: None,
                orbit_yaw_deg: Some(opts.yaw_deg),
                orbit_pitch_deg: Some(opts.pitch_deg),
                orbit_distance: Some(opts.distance),
            },
        },
    });

    let targets: Vec<SweepTarget> = if opts.targets.is_empty() {
        pick_targets(world, opts.zoom_views)
            .into_iter()
            .enumerate()
            .map(|(i, p)| SweepTarget {
                id: format!("zoom{}", i + 1),
                latitude_deg: p.y.clamp(-1.0, 1.0).asin().to_degrees(),
                longitude_deg: p.z.atan2(p.x).to_degrees(),
            })
            .collect()
    } else {
        opts.targets.clone()
    };
    for target in targets {
        let center = target.unit_position();
        let (vp, eye) = target_camera(center, aspect, opts.zoom_alt);
        let n = center.normalize();
        let aim = n * 1.08;
        views.push(CaptureView {
            view_proj: vp,
            eye,
            label: target.id.clone(),
            sidecar: ViewRecord {
                id: target.id.clone(),
                kind: if opts.targets.is_empty() {
                    "automatic-highland"
                } else {
                    "explicit-dossier-target"
                },
                target: Some(target),
                camera: CameraRecord {
                    eye_xyz: vec3_array(eye),
                    aim_xyz: vec3_array(aim),
                    up_xyz: vec3_array(n),
                    vertical_fov_deg: 45.0,
                    aspect,
                    near: 0.01,
                    far: 10.0,
                    target_altitude: Some(opts.zoom_alt),
                    orbit_yaw_deg: None,
                    orbit_pitch_deg: None,
                    orbit_distance: None,
                },
            },
        });
    }
    views
}

/// Cumulative-stack presets: each tile layers one rung's knobs on the previous, so
/// the rows read as "as if I'd signed off on each in turn". The `base_erosion`
/// supplies the non-P1 knobs; the P1 knobs are set explicitly per tile.
fn build_stack_tiles(
    name: &str,
    base: &ErosionOverrides,
) -> Vec<(ErosionOverrides, String, String)> {
    let tile = |ir: f32, fsw: f32, mc: f32, label: &str, fname: &str| {
        let mut o = *base;
        o.interior_relief = Some(ir);
        o.front_strike_weight = Some(fsw);
        o.margin_contrast = Some(mc);
        (o, label.to_string(), fname.to_string())
    };
    // erosion-v3: an emergent-build tile — demote λ, rebuild via active uplift
    // (uplift_scale calibrated ≈ λ/(steps·dt)), with the channelization machinery on
    // (MFD + nonlinear hillslope), faint seed, P1b/c off.
    // Emergent build (erosion-v3): demote λ, rebuild via the SELF-CALIBRATING builder
    // (uplift auto-derived from the demotion, so height tracks target and `steps` is a
    // pure build-vs-carve dial). Seed-only base (no painted strike/margin/scarps — the
    // point is erosion BUILDS the relief), channelization machinery on.
    let emergent = |lambda: f32, steps: usize, label: &str, fname: &str| {
        let mut o = *base;
        o.emergent_lambda = Some(lambda);
        o.interior_relief = Some(0.005); // faint seed
        o.front_strike_weight = Some(0.0);
        o.margin_contrast = Some(0.0);
        o.fault_scarp_height = Some(0.0);
        o.steps = Some(steps);
        o.mfd_exponent = Some(1.0);
        o.hillslope_critical_slope = Some(200.0);
        (o, label.to_string(), fname.to_string())
    };
    // O0 (orogen-structure): emergent build at n=2 with optional STRUCTURED uplift
    // (asymmetric + segmented) — the decisive A/B for whether tectonic uplift beats the
    // smooth-emergent dome. Seed-only, P1b/c off, more carving steps.
    let emergent_o0 = |structured: f32, label: &str, fname: &str| {
        let mut o = *base;
        o.emergent_lambda = Some(0.5);
        o.emergent_structured = Some(structured);
        o.interior_relief = Some(0.005);
        o.front_strike_weight = Some(0.0);
        o.margin_contrast = Some(0.0);
        o.fault_scarp_height = Some(0.0);
        o.steps = Some(200);
        o.n = Some(2.0);
        o.hillslope_critical_slope = Some(200.0);
        (o, label.to_string(), fname.to_string())
    };
    match name {
        // O3A architecture probe: identical product erosion over three
        // pre-hydrology substrates. The isotropic amplitude was measured on
        // seed 12345/100k/250k to match the MassifCorridor arm's 44 m
        // area-weighted structural RMS over the fixed tectonic footprint.
        "o3a" => {
            let arm = |interior: f32, meso_base: f32, label: &str, fname: &str| {
                let mut o = *base;
                o.fault_scarp_height = Some(0.0);
                o.interior_relief = Some(interior);
                o.front_strike_weight = Some(0.0);
                o.margin_contrast = Some(0.0);
                o.emergent_lambda = Some(0.0);
                o.emergent_structured = Some(0.0);
                o.meso_relief = Some(0.0);
                o.meso_base_relief = Some(meso_base);
                o.meso_style = Some(1);
                o.meso_wavelength_km = Some(25.0);
                (o, label.to_string(), fname.to_string())
            };
            vec![
                arm(0.0, 0.0, "legacy substrate (0 m RMS)", "0_legacy"),
                arm(0.0611, 0.0, "isotropic P1a (44 m RMS)", "1_isotropic_44m"),
                arm(
                    0.0,
                    0.05,
                    "MassifCorridor base (44 m RMS)",
                    "2_massif_corridor_44m",
                ),
            ]
        }
        // erosion-v2 Phase 1: flat interpolant → +interior grain → +strike → +margin.
        "p1" => vec![
            tile(0.0, 0.0, 0.0, "baseline (P1 off)", "0_baseline"),
            tile(0.04, 0.0, 0.0, "P1a interior", "1_p1a"),
            tile(0.04, 0.7, 0.0, "P1a+P1b strike", "2_p1ab"),
            tile(0.04, 0.7, 1.0, "P1a+P1b+P1c margin", "3_p1abc"),
        ],
        // erosion-v3: painted P1 (current best) vs emergent build at λ=0.5, sweeping
        // STEPS (the build-vs-carve budget) — does dissection emerge with more carving?
        "v3" => vec![
            tile(0.04, 0.7, 1.0, "P1 painted (current)", "0_painted"),
            emergent(0.5, 120, "emergent λ=0.5, 120 steps", "1_emergent_s120"),
            emergent(0.5, 240, "emergent λ=0.5, 240 steps", "2_emergent_s240"),
            emergent(0.5, 400, "emergent λ=0.5, 400 steps", "3_emergent_s400"),
        ],
        // O0: painted vs smooth-emergent vs structured-emergent (all n=2). Does the
        // asymmetric+segmented tectonic uplift read as RANGES vs the smooth dome?
        "o0" => vec![
            tile(0.04, 0.7, 1.0, "P1 painted (current)", "0_painted"),
            emergent_o0(0.0, "emergent smooth (n=2)", "1_emergent_smooth"),
            emergent_o0(1.0, "emergent STRUCTURED (n=2)", "2_emergent_structured"),
        ],
        // Meso default-flip A/B (spec §13 addenda): the gain dial is retired from
        // the meso path (peaks "reach the heavens" — user 2026-07-10; plausibility
        // self-gate now vetoes >12 km before visual). Candidate = corridor-heavy
        // field at gain 1: relief from valleys down at a fixed peak budget.
        // Labels = 25-km p95-p05 p50 relief / max range peak / self-gate verdict.
        "meso" => {
            let meso = |relief: f32, label: &str, fname: &str| {
                let mut o = *base;
                o.meso_relief = Some(relief);
                o.meso_style = Some(1);
                o.steps = Some(50);
                (o, label.to_string(), fname.to_string())
            };
            vec![
                (
                    *base,
                    "baseline (191 m, peaks 10.0 km)".to_string(),
                    "0_baseline".to_string(),
                ),
                meso(0.7, "CANDIDATE m0.7 s50 (313 m, 11.8 km, ok)", "1_cand_m07"),
                meso(0.9, "m0.9 s50 (362 m, 12.4 km, borderline)", "2_max_m09"),
            ]
        }
        other => panic!("unknown --sweep-stack '{other}'; known: o3a, p1, v3, o0, meso"),
    }
}

#[derive(Debug, Serialize)]
struct RangeAncestrySidecar {
    schema_version: u32,
    purpose: &'static str,
    coordinate_convention: &'static str,
    color_sampling: &'static str,
    world_manifest: hex3::world::RunManifest,
    cameras: Vec<ViewRecord>,
    layers: Vec<RangeLayerRecord>,
    montage_filename: &'static str,
}

#[derive(Debug, Serialize)]
struct RangeLayerRecord {
    index: usize,
    id: &'static str,
    label: &'static str,
    topology: &'static str,
    role: &'static str,
    source: &'static str,
    units: &'static str,
    relief_scale: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    robust_color_min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    robust_color_max: Option<f32>,
    image_filenames: Vec<String>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct RoofCompilerCounterfactualSidecar {
    schema_version: u32,
    purpose: &'static str,
    status: &'static str,
    coordinate_convention: &'static str,
    color_sampling: &'static str,
    world_manifest: hex3::world::RunManifest,
    counterfactuals: RoofCompilerCounterfactualContract,
    cameras: Vec<ViewRecord>,
    layers: Vec<RangeLayerRecord>,
    montage_filename: &'static str,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct RoofCompilerCounterfactualContract {
    baseline_coarse_elevation: &'static str,
    baseline_collision_response: &'static str,
    reconstruction_formula: &'static str,
    nearest_source_response: &'static str,
    episode_mean_response: &'static str,
    interpolation_support: &'static str,
    interpolation_weight: &'static str,
    interpolation_epsilon: f32,
    baseline_reinterpolation_max_abs_error: f32,
    work_matching_measure: &'static str,
    nearest_source_work_scale: f32,
    episode_mean_work_scale: f32,
    erosion_policy: &'static str,
}

struct DiagnosticMeshBuffers {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

fn robust_scale(values: &[f32]) -> (f32, f32) {
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    assert!(!finite.is_empty(), "diagnostic layer has no finite values");
    finite.sort_unstable_by(f32::total_cmp);
    let quantile = |q: f32| {
        let i = (q * (finite.len() - 1) as f32).round() as usize;
        finite[i]
    };
    let lo = quantile(0.02);
    let mut hi = quantile(0.98);
    if hi <= lo {
        hi = finite[finite.len() - 1];
    }
    if hi <= lo {
        hi = lo + 1.0;
    }
    (lo, hi)
}

fn scalar_gray(value: f32, lo: f32, hi: f32) -> Vec3 {
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    // Keep both ends away from black/white so lighting and mesh facets remain legible.
    Vec3::splat(0.12 + 0.76 * t)
}

fn diagnostic_mesh(
    device: &wgpu::Device,
    tess: &Tessellation,
    elevation: &[f32],
    color_values: Option<(&[f32], f32, f32)>,
) -> DiagnosticMeshBuffers {
    assert_eq!(tess.num_cells(), elevation.len());
    if let Some((values, _, _)) = color_values {
        assert_eq!(tess.num_cells(), values.len());
    }
    let color = |i: usize| match color_values {
        Some((values, lo, hi)) => scalar_gray(values[i], lo, hi),
        None => Vec3::splat(0.52),
    };
    // Use the same shared-vertex interpolation for coarse and fine layers. A
    // per-cell coarse mesh would introduce flat facets absent from the fine rows
    // and confound the ancestry comparison. Scalar colors are consequently
    // vertex-averaged too, providing topology-consistent anti-aliasing.
    let mesh = UnifiedMesh::from_voronoi_shared_vertices(
        &tess.voronoi,
        color,
        |_| Material::Land,
        |i| elevation[i].max(0.0),
    );
    DiagnosticMeshBuffers {
        vertices: create_vertex_buffer(device, &mesh.vertices, "range_ancestry_vertices"),
        indices: create_index_buffer(device, &mesh.indices, "range_ancestry_indices"),
        index_count: mesh.indices.len() as u32,
    }
}

#[allow(clippy::too_many_arguments)]
fn render_diagnostic(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_view: &wgpu::TextureView,
    buffers: &DiagnosticMeshBuffers,
    river_bind_group: &wgpu::BindGroup,
    view_proj: Mat4,
    cam_pos: Vec3,
    relief_scale: f32,
    slope_shading: bool,
) {
    let uniforms = Uniforms::new(view_proj, cam_pos, Vec3::new(0.5, 1.0, 0.3).normalize())
        .with_relief_scale(relief_scale)
        .with_slope_shading(slope_shading)
        .with_hemisphere_lighting(false)
        .with_map_mode(false)
        .with_rivers(false);
    renderer.render_to_view(
        &gpu.device,
        &gpu.queue,
        color_view,
        &uniforms,
        RenderScene {
            fill_pipeline: FillPipelineKind::UnifiedGlobe,
            fill: IndexedDraw {
                vertex_buffer: &buffers.vertices,
                index_buffer: &buffers.indices,
                index_count: buffers.index_count,
            },
            river_texture_bind_group: Some(river_bind_group),
            edges: None,
            arrows: None,
            pole_markers: None,
            rivers: None,
            gpu_particles: None,
        },
    );
}

struct WaterGeographyBuffers {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    lines: wgpu::Buffer,
    line_count: u32,
}

#[derive(Debug, Serialize)]
struct WaterGeographyPacketSidecar {
    schema_version: u32,
    purpose: &'static str,
    coordinate_convention: &'static str,
    geometry_contract: &'static str,
    world_manifest: hex3::world::RunManifest,
    cameras: Vec<ViewRecord>,
    coast_selection: CoastSelectionRecord,
    spill_selection: SpillSelectionRecord,
    topology: ShorelineTopologyRecord,
    layers: Vec<WaterGeographyLayerRecord>,
    colors: WaterGeographyColorRecord,
    montage_filename: &'static str,
}

#[derive(Debug, Serialize)]
struct CoastSelectionRecord {
    rule: &'static str,
    fallback_used: bool,
    ocean_water_body_id: WaterBodyId,
    loop_anchor_edges: Vec<[u32; 2]>,
    landmass_anchor_cells: Vec<usize>,
    loop_lengths_km: Vec<f32>,
    nearest_sample_distance_km: f32,
}

#[derive(Debug, Serialize)]
struct SpillSelectionRecord {
    rule: &'static str,
    fallback_used: bool,
    basin_id: usize,
    water_body_id: Option<WaterBodyId>,
    currently_overflowing: bool,
    destination: BasinSpillDestination,
    route_cell_count: usize,
    integration_cut_cell_count: usize,
}

#[derive(Debug, Serialize)]
struct ShorelineTopologyRecord {
    loop_count: usize,
    edge_count: usize,
    unresolved_edge_count: usize,
    issue_count: usize,
}

#[derive(Debug, Serialize)]
struct WaterGeographyLayerRecord {
    id: &'static str,
    role: &'static str,
    relief_scale: f32,
    image_filenames: Vec<String>,
}

#[derive(Debug, Serialize)]
struct WaterGeographyColorRecord {
    ocean_shoreline: &'static str,
    lake_shoreline: &'static str,
    selected_coast_a: &'static str,
    selected_coast_b: &'static str,
    spill_route: &'static str,
    integration_cut_route: &'static str,
    unresolved_shoreline: &'static str,
}

struct CoastSelection {
    loop_indices: Vec<usize>,
    target: Vec3,
    nearest_distance_km: f32,
    fallback_used: bool,
}

struct SpillSelection {
    route_index: usize,
    water_body_index: Option<usize>,
    target: Vec3,
    fallback_used: bool,
}

fn sampled_loop_positions<'a>(
    tess: &'a Tessellation,
    shoreline: &'a ShorelineLoop,
) -> impl Iterator<Item = Vec3> + 'a {
    let stride = shoreline.edges.len().div_ceil(128).max(1);
    shoreline
        .edges
        .iter()
        .step_by(stride)
        .map(|edge| tess.voronoi.vertices[edge.from_vertex as usize])
}

fn nearest_loop_sample_pair(
    tess: &Tessellation,
    a: &ShorelineLoop,
    b: &ShorelineLoop,
) -> (Vec3, Vec3, f32) {
    let a_positions: Vec<Vec3> = sampled_loop_positions(tess, a).collect();
    let b_positions: Vec<Vec3> = sampled_loop_positions(tess, b).collect();
    let mut best = (a_positions[0], b_positions[0], -1.0f32);
    for &pa in &a_positions {
        for &pb in &b_positions {
            let dot = pa.dot(pb);
            if dot > best.2 {
                best = (pa, pb, dot);
            }
        }
    }
    let distance_km = (best.0 - best.1).length() * hex3::world::PLANET_RADIUS_KM;
    (best.0, best.1, distance_km)
}

fn select_coast_complex(tess: &Tessellation, geometry: &WaterGeographyGeometry) -> CoastSelection {
    const MIN_COMPLEX_LOOP_LENGTH_KM: f32 = 500.0;
    let loops = &geometry.shoreline.loops;
    let ocean_indices: Vec<usize> = loops
        .iter()
        .enumerate()
        .filter(|(_, shoreline)| shoreline.water_kind == SemanticWaterKind::Ocean)
        .map(|(index, _)| index)
        .collect();
    assert!(
        !ocean_indices.is_empty(),
        "water-geography packet requires an ocean coast"
    );

    let mut best: Option<(usize, usize, Vec3, Vec3, f32, f32)> = None;
    for (position, &a_index) in ocean_indices.iter().enumerate() {
        let a = &loops[a_index];
        for &b_index in &ocean_indices[position + 1..] {
            let b = &loops[b_index];
            if a.water_body_id != b.water_body_id
                || a.adjacent_landmass_anchor_cells == b.adjacent_landmass_anchor_cells
                || a.length_km < MIN_COMPLEX_LOOP_LENGTH_KM
                || b.length_km < MIN_COMPLEX_LOOP_LENGTH_KM
            {
                continue;
            }
            let (pa, pb, distance) = nearest_loop_sample_pair(tess, a, b);
            let scale = a.length_km.min(b.length_km);
            let replace = best
                .as_ref()
                .map(|current| distance < current.4 || (distance == current.4 && scale > current.5))
                .unwrap_or(true);
            if replace {
                best = Some((a_index, b_index, pa, pb, distance, scale));
            }
        }
    }
    if let Some((a, b, pa, pb, distance, _)) = best {
        return CoastSelection {
            loop_indices: vec![a, b],
            target: (pa + pb).normalize_or_zero(),
            nearest_distance_km: distance,
            fallback_used: false,
        };
    }

    let &index = ocean_indices
        .iter()
        .max_by(|&&a, &&b| loops[a].length_km.total_cmp(&loops[b].length_km))
        .unwrap();
    let edge = loops[index].edges[0];
    let from = tess.voronoi.vertices[edge.from_vertex as usize];
    let to = tess.voronoi.vertices[edge.to_vertex as usize];
    CoastSelection {
        loop_indices: vec![index],
        target: (from + to).normalize(),
        nearest_distance_km: 0.0,
        fallback_used: true,
    }
}

fn select_spill(
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    water: &WaterBodySemantics,
    geometry: &WaterGeographyGeometry,
) -> SpillSelection {
    let selected_lake = water
        .bodies
        .iter()
        .enumerate()
        .filter(|(_, body)| body.kind == SemanticWaterKind::Lake)
        .filter_map(|(body_index, body)| {
            let basin_id = body.id.basin_id?;
            let basin = hydrology.basins.get(basin_id)?;
            (basin.has_water() && basin.is_overflowing()).then_some((body_index, body.area_km2))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));

    if let Some((body_index, _)) = selected_lake {
        let body = &water.bodies[body_index];
        let basin_id = body.id.basin_id.unwrap();
        let route_index = geometry
            .basin_spill_routes
            .iter()
            .position(|route| route.basin_id == basin_id)
            .expect("selected lake basin has spill route geometry");
        return SpillSelection {
            route_index,
            water_body_index: Some(body_index),
            target: tess.cell_center(body.id.anchor_cell),
            fallback_used: false,
        };
    }

    let (route_index, route) = geometry
        .basin_spill_routes
        .iter()
        .enumerate()
        .filter(|(_, route)| !route.cells.is_empty())
        .max_by_key(|(_, route)| route.cells.len())
        .expect("water-geography packet requires at least one spill route");
    SpillSelection {
        route_index,
        water_body_index: None,
        target: tess.cell_center(route.cells[0]),
        fallback_used: true,
    }
}

fn derived_capture_view(id: &str, center: Vec3, aspect: f32, altitude: f32) -> CaptureView {
    let n = center.normalize();
    // The ordinary close-up helper aims at radius 1.08 to frame exaggerated
    // mountains. This packet must also show a flat semantic sphere with the exact
    // same camera, so aim near the physical surface instead of pushing it to the
    // limb in the diagnostic row.
    let aim = n * 1.02;
    let up_ref = if n.y.abs() < 0.95 { Vec3::Y } else { Vec3::Z };
    let east = n.cross(up_ref).normalize();
    let north = east.cross(n).normalize();
    let eye = aim + n * altitude + north * (altitude * 0.55);
    let view = Mat4::look_at_rh(eye, aim, n);
    let projection = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 10.0);
    let view_proj = projection * view;
    let target = SweepTarget {
        id: id.to_string(),
        latitude_deg: center.y.clamp(-1.0, 1.0).asin().to_degrees(),
        longitude_deg: center.z.atan2(center.x).to_degrees(),
    };
    CaptureView {
        view_proj,
        eye,
        label: id.to_string(),
        sidecar: ViewRecord {
            id: id.to_string(),
            kind: "derived-water-geography",
            target: Some(target),
            camera: CameraRecord {
                eye_xyz: vec3_array(eye),
                aim_xyz: vec3_array(aim),
                up_xyz: vec3_array(n),
                vertical_fov_deg: 45.0,
                aspect,
                near: 0.01,
                far: 10.0,
                target_altitude: Some(altitude),
                orbit_yaw_deg: None,
                orbit_pitch_deg: None,
                orbit_distance: None,
            },
        },
    }
}

fn ownership_color(
    cell: usize,
    geometry: &WaterGeographyGeometry,
    hydrology: &hex3::world::Hydrology,
    water: &WaterBodySemantics,
    coast_landmasses: &[usize],
    spill_body_index: Option<usize>,
) -> Vec3 {
    if hydrology.is_submerged(cell) {
        let body_index = water.cell_body[cell].expect("submerged cell has semantic owner");
        let body = &water.bodies[body_index];
        return match body.kind {
            SemanticWaterKind::Ocean => Vec3::new(0.05, 0.14, 0.27),
            SemanticWaterKind::Lake if Some(body_index) == spill_body_index => {
                Vec3::new(0.05, 0.75, 0.78)
            }
            SemanticWaterKind::Lake => Vec3::new(0.08, 0.38, 0.50),
            SemanticWaterKind::Pond => Vec3::new(0.65, 0.45, 0.10),
        };
    }
    let anchor = geometry.shoreline.landmass_anchor_by_cell[cell].unwrap_or(cell);
    if coast_landmasses.first() == Some(&anchor) {
        return Vec3::new(0.95, 0.58, 0.12);
    }
    if coast_landmasses.get(1) == Some(&anchor) {
        return Vec3::new(0.45, 0.85, 0.24);
    }
    let hash = (anchor as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let tint = ((hash >> 56) as f32) / 255.0;
    Vec3::new(0.30 + 0.18 * tint, 0.34 + 0.12 * (1.0 - tint), 0.26)
}

fn water_geography_buffers(
    device: &wgpu::Device,
    world: &World,
    water: &WaterBodySemantics,
    geometry: &WaterGeographyGeometry,
    coast: &CoastSelection,
    spill: &SpillSelection,
) -> WaterGeographyBuffers {
    let tess = world.active_tessellation();
    let hydrology = world.active_hydrology().expect("water geography hydrology");
    let selected_loops: std::collections::BTreeSet<usize> =
        coast.loop_indices.iter().copied().collect();
    let mut coast_landmasses = Vec::new();
    for &loop_index in &coast.loop_indices {
        coast_landmasses.extend(
            geometry.shoreline.loops[loop_index]
                .adjacent_landmass_anchor_cells
                .iter()
                .copied(),
        );
    }
    coast_landmasses.sort_unstable();
    coast_landmasses.dedup();
    let mesh = VoronoiMesh::from_voronoi_with_colors(&tess.voronoi, |cell| {
        ownership_color(
            cell,
            geometry,
            hydrology,
            water,
            &coast_landmasses,
            spill.water_body_index,
        )
    });

    let mut lines = Vec::new();
    for (loop_index, shoreline) in geometry.shoreline.loops.iter().enumerate() {
        let color = if selected_loops.contains(&loop_index) {
            if coast.loop_indices.first() == Some(&loop_index) {
                Vec3::new(1.0, 0.92, 0.20)
            } else {
                Vec3::new(0.95, 0.25, 0.85)
            }
        } else if shoreline.water_kind == SemanticWaterKind::Ocean {
            Vec3::new(0.15, 0.65, 0.95)
        } else {
            Vec3::new(0.95, 0.62, 0.15)
        };
        for edge in &shoreline.edges {
            lines.push(SurfaceVertex::new(
                tess.voronoi.vertices[edge.from_vertex as usize],
                0.0,
                color,
                1.0,
            ));
            lines.push(SurfaceVertex::new(
                tess.voronoi.vertices[edge.to_vertex as usize],
                0.0,
                color,
                1.0,
            ));
        }
    }
    for unresolved in &geometry.shoreline.unresolved_edges {
        let edge = unresolved.edge;
        for vertex in [edge.from_vertex, edge.to_vertex] {
            lines.push(SurfaceVertex::new(
                tess.voronoi.vertices[vertex as usize],
                0.0,
                Vec3::new(1.0, 0.05, 0.05),
                1.0,
            ));
        }
    }
    let route = &geometry.basin_spill_routes[spill.route_index];
    for pair in route.cells.windows(2) {
        let cut = hydrology.was_lowered_by_integration(pair[0])
            || hydrology.was_lowered_by_integration(pair[1]);
        let color = if cut {
            Vec3::new(1.0, 0.08, 0.05)
        } else {
            Vec3::new(0.95, 0.10, 0.95)
        };
        for &cell in pair {
            lines.push(SurfaceVertex::new(tess.cell_center(cell), 0.0, color, 1.0));
        }
    }

    WaterGeographyBuffers {
        vertices: create_vertex_buffer(device, &mesh.vertices, "water_geography_vertices"),
        indices: create_index_buffer(device, &mesh.indices, "water_geography_indices"),
        index_count: mesh.indices.len() as u32,
        lines: create_vertex_buffer(device, &lines, "water_geography_lines"),
        line_count: lines.len() as u32,
    }
}

fn render_water_geography(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_view: &wgpu::TextureView,
    buffers: &WaterGeographyBuffers,
    view: &CaptureView,
) {
    let uniforms = Uniforms::new(
        view.view_proj,
        view.eye,
        Vec3::new(0.5, 1.0, 0.3).normalize(),
    )
    .with_relief_scale(0.0)
    .with_hemisphere_lighting(false)
    .with_map_mode(false)
    .with_rivers(false);
    renderer.render_to_view(
        &gpu.device,
        &gpu.queue,
        color_view,
        &uniforms,
        RenderScene {
            fill_pipeline: FillPipelineKind::Globe,
            fill: IndexedDraw {
                vertex_buffer: &buffers.vertices,
                index_buffer: &buffers.indices,
                index_count: buffers.index_count,
            },
            river_texture_bind_group: None,
            edges: None,
            arrows: None,
            pole_markers: None,
            rivers: Some(SurfaceLineDraw {
                vertex_buffer: &buffers.lines,
                vertex_count: buffers.line_count,
            }),
            gpu_particles: None,
        },
    );
}

fn run_water_geography_packet(opts: &SweepOptions) {
    assert_eq!(
        opts.target_stage, 4,
        "water-geography packet requires --stage 4"
    );
    let world = generate_tile_world(opts, &opts.base_erosion);
    let tess = world.active_tessellation();
    let hydrology = world.active_hydrology().expect("stage 4 hydrology");
    let water = WaterBodySemantics::build(tess, hydrology);
    let geometry = WaterGeographyGeometry::build(tess, hydrology, &water)
        .expect("derive exact water-geography geometry");
    let coast = select_coast_complex(tess, &geometry);
    let spill = select_spill(tess, hydrology, &water, &geometry);
    let aspect = opts.width as f32 / opts.height as f32;
    let views = [
        derived_capture_view("coast-complex", coast.target, aspect, opts.zoom_alt),
        derived_capture_view("lake-spill", spill.target, aspect, opts.zoom_alt),
    ];

    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|e| panic!("create {}: {e}", opts.out_dir.display()));
    let gpu = pollster::block_on(GpuContext::new_headless(opts.width, opts.height));
    let mut renderer = Renderer::new(&gpu, &Uniforms::new(Mat4::IDENTITY, Vec3::ZERO, Vec3::Y));
    let color_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("water_geography_color"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&Default::default());
    let montage_w = opts.width * 2;
    let montage_h = opts.height * 2;
    let mut montage = vec![0; (montage_w * montage_h * 4) as usize];

    let physical_buffers = generate_world_buffers(&gpu.device, &gpu.queue, &world);
    let mut physical_files = Vec::new();
    for (column, view) in views.iter().enumerate() {
        render_relief(
            &gpu,
            &mut renderer,
            &color_view,
            &physical_buffers,
            view.view_proj,
            view.eye,
            opts.river_mode,
            RELIEF_SCALE,
            1.0,
        );
        let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
        let filename = format!("01_physical_{}.png", view.label);
        write_png(
            &opts.out_dir.join(&filename),
            &rgba,
            opts.width,
            opts.height,
        );
        blit_tile(
            &mut montage,
            montage_w,
            &rgba,
            opts.width,
            opts.height,
            column as u32,
            0,
        );
        physical_files.push(filename);
    }
    drop(physical_buffers);

    let diagnostic_buffers =
        water_geography_buffers(&gpu.device, &world, &water, &geometry, &coast, &spill);
    let mut diagnostic_files = Vec::new();
    for (column, view) in views.iter().enumerate() {
        render_water_geography(&gpu, &mut renderer, &color_view, &diagnostic_buffers, view);
        let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
        let filename = format!("02_diagnostic_{}.png", view.label);
        write_png(
            &opts.out_dir.join(&filename),
            &rgba,
            opts.width,
            opts.height,
        );
        blit_tile(
            &mut montage,
            montage_w,
            &rgba,
            opts.width,
            opts.height,
            column as u32,
            1,
        );
        diagnostic_files.push(filename);
    }
    write_png(
        &opts.out_dir.join("montage.png"),
        &montage,
        montage_w,
        montage_h,
    );

    let selected_loops: Vec<&ShorelineLoop> = coast
        .loop_indices
        .iter()
        .map(|&index| &geometry.shoreline.loops[index])
        .collect();
    let selected_route = &geometry.basin_spill_routes[spill.route_index];
    let selected_basin = &hydrology.basins[selected_route.basin_id];
    let selected_water_body = spill.water_body_index.map(|index| water.bodies[index].id);
    let sidecar = WaterGeographyPacketSidecar {
        schema_version: 1,
        purpose: "matched physical and semantic views of coastline ownership and a basin spill route",
        coordinate_convention: "latitude=asin(y); longitude=atan2(z,x); degrees; positive longitude rotates +X toward +Z",
        geometry_contract: "exact boundaries of categorical ocean/lake Voronoi-cell masks; water is left of every directed edge; no contour interpolation or cartographic simplification",
        world_manifest: world.manifest(),
        cameras: views.iter().map(|view| view.sidecar.clone()).collect(),
        coast_selection: CoastSelectionRecord {
            rule: "nearest sampled pair of raw ocean shoreline loops at least 500 km long, belonging to distinct landmasses but the same semantic ocean; largest ocean loop fallback",
            fallback_used: coast.fallback_used,
            ocean_water_body_id: selected_loops[0].water_body_id,
            loop_anchor_edges: selected_loops.iter().map(|shore| shore.anchor_edge).collect(),
            landmass_anchor_cells: selected_loops
                .iter()
                .flat_map(|shore| shore.adjacent_landmass_anchor_cells.iter().copied())
                .collect(),
            loop_lengths_km: selected_loops.iter().map(|shore| shore.length_km).collect(),
            nearest_sample_distance_km: coast.nearest_distance_km,
        },
        spill_selection: SpillSelectionRecord {
            rule: "largest-area wet overflowing semantic lake; longest available potential basin route fallback",
            fallback_used: spill.fallback_used,
            basin_id: selected_route.basin_id,
            water_body_id: selected_water_body,
            currently_overflowing: selected_basin.is_overflowing(),
            destination: selected_route.destination.clone(),
            route_cell_count: selected_route.cells.len(),
            integration_cut_cell_count: selected_route
                .cells
                .iter()
                .filter(|&&cell| hydrology.was_lowered_by_integration(cell))
                .count(),
        },
        topology: ShorelineTopologyRecord {
            loop_count: geometry.shoreline.loops.len(),
            edge_count: geometry
                .shoreline
                .loops
                .iter()
                .map(|shore| shore.edges.len())
                .sum(),
            unresolved_edge_count: geometry.shoreline.unresolved_edges.len(),
            issue_count: geometry.shoreline.issues.len(),
        },
        layers: vec![
            WaterGeographyLayerRecord {
                id: "physical",
                role: "authentic final relief with ordinary river presentation",
                relief_scale: RELIEF_SCALE,
                image_filenames: physical_files,
            },
            WaterGeographyLayerRecord {
                id: "diagnostic",
                role: "flat categorical ownership plus exact raw shorelines and selected spill provenance",
                relief_scale: 0.0,
                image_filenames: diagnostic_files,
            },
        ],
        colors: WaterGeographyColorRecord {
            ocean_shoreline: "#26a6f2",
            lake_shoreline: "#f29e26",
            selected_coast_a: "#ffeb33",
            selected_coast_b: "#f240d9",
            spill_route: "#f21af2",
            integration_cut_route: "#ff140d",
            unresolved_shoreline: "#ff0d0d",
        },
        montage_filename: "montage.png",
    };
    let sidecar_file = std::fs::File::create(opts.out_dir.join("water-geography.json"))
        .expect("create water-geography.json");
    serde_json::to_writer_pretty(BufWriter::new(sidecar_file), &sidecar)
        .expect("write water-geography.json");
    println!("Done: water-geography packet -> {}", opts.out_dir.display());
}

#[derive(Clone, Copy)]
enum LivingPreviewLayer {
    Heat,
    RelativeWater,
    DrainageSaturation,
    Vegetation,
    Bare,
    Herbaceous,
    Woody,
    Wetland,
}

impl LivingPreviewLayer {
    const ALL: [Self; 8] = [
        Self::Heat,
        Self::RelativeWater,
        Self::DrainageSaturation,
        Self::Vegetation,
        Self::Bare,
        Self::Herbaceous,
        Self::Woody,
        Self::Wetland,
    ];

    fn id(self) -> &'static str {
        match self {
            Self::Heat => "thermal-opportunity",
            Self::RelativeWater => "climatic-water-availability",
            Self::DrainageSaturation => "drainage-saturation",
            Self::Vegetation => "vegetation-cover",
            Self::Bare => "bare-fraction",
            Self::Herbaceous => "herbaceous-fraction",
            Self::Woody => "woody-fraction",
            Self::Wetland => "wetland-fraction",
        }
    }

    fn role(self) -> &'static str {
        match self {
            Self::Heat | Self::RelativeWater | Self::DrainageSaturation => {
                "prototype input or limiting factor"
            }
            Self::Vegetation => "continuous cover consequence",
            Self::Bare | Self::Herbaceous | Self::Woody | Self::Wetland => {
                "exclusive physiognomy fraction"
            }
        }
    }
}

fn mix_color(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a.lerp(b, t.clamp(0.0, 1.0))
}

fn living_preview_color(
    layer: LivingPreviewLayer,
    cell: &hex3::world::LivingSurfaceCell,
    ocean: bool,
    submerged: bool,
) -> Vec3 {
    if ocean {
        return Vec3::new(0.05, 0.12, 0.20);
    }
    if submerged {
        return Vec3::new(0.08, 0.30, 0.52);
    }
    match layer {
        LivingPreviewLayer::Heat => mix_color(
            Vec3::new(0.18, 0.34, 0.72),
            Vec3::new(0.96, 0.72, 0.18),
            cell.thermal_opportunity,
        ),
        LivingPreviewLayer::RelativeWater => mix_color(
            Vec3::new(0.55, 0.27, 0.10),
            Vec3::new(0.10, 0.55, 0.42),
            1.0 - cell.relative_water_limitation,
        ),
        LivingPreviewLayer::DrainageSaturation => mix_color(
            Vec3::new(0.34, 0.30, 0.20),
            Vec3::new(0.06, 0.74, 0.92),
            cell.drainage_saturation,
        ),
        LivingPreviewLayer::Vegetation => mix_color(
            Vec3::new(0.50, 0.35, 0.18),
            Vec3::new(0.16, 0.66, 0.22),
            cell.vegetation_cover,
        ),
        LivingPreviewLayer::Bare => mix_color(
            Vec3::new(0.20, 0.42, 0.20),
            Vec3::new(0.72, 0.50, 0.24),
            cell.fractions.bare,
        ),
        LivingPreviewLayer::Herbaceous => mix_color(
            Vec3::new(0.45, 0.38, 0.22),
            Vec3::new(0.55, 0.72, 0.18),
            cell.fractions.herbaceous,
        ),
        LivingPreviewLayer::Woody => mix_color(
            Vec3::new(0.48, 0.40, 0.22),
            Vec3::new(0.04, 0.34, 0.09),
            cell.fractions.woody,
        ),
        LivingPreviewLayer::Wetland => mix_color(
            Vec3::new(0.42, 0.35, 0.23),
            Vec3::new(0.12, 0.68, 0.60),
            cell.fractions.wetland,
        ),
    }
}

fn living_presentation_mesh<F>(
    device: &wgpu::Device,
    world: &World,
    color_fn: F,
) -> DiagnosticMeshBuffers
where
    F: Fn(usize) -> Vec3,
{
    let tess = world.active_tessellation();
    let elevation = world.active_elevation().expect("stage 4 elevation");
    let hydrology = world.active_hydrology().expect("stage 4 hydrology");
    let mesh = UnifiedMesh::from_voronoi_shared_vertices(
        &tess.voronoi,
        color_fn,
        |cell| cell_material(world, cell),
        |cell| {
            if hydrology.is_ocean(cell) {
                0.0
            } else if hydrology.is_lake_water(cell) {
                hydrology
                    .basin(cell)
                    .map(|basin| basin.water_level)
                    .unwrap_or(0.0)
            } else {
                elevation.values[cell].max(0.0)
            }
        },
    );
    DiagnosticMeshBuffers {
        vertices: create_vertex_buffer(device, &mesh.vertices, "living_presentation_vertices"),
        indices: create_index_buffer(device, &mesh.indices, "living_presentation_indices"),
        index_count: mesh.indices.len() as u32,
    }
}

/// The unified globe pipeline always requires its river texture group, even
/// when rivers are disabled. A transparent texel avoids constructing the full
/// product river texture and keeps this diagnostic bounded.
fn transparent_river_binding(gpu: &GpuContext) -> (wgpu::Texture, wgpu::BindGroup) {
    let texture = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some("living_surface_transparent_river"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
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
        &[0, 0, 0, 0],
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("living_surface_transparent_river_sampler"),
        ..Default::default()
    });
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("living_surface_transparent_river_bind_group"),
        layout: &hex3::render::create_river_bind_group_layout(&gpu.device),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    (texture, bind_group)
}

fn render_living_preview_map(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_view: &wgpu::TextureView,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    index_count: u32,
) {
    // The map shader normalizes both longitude and latitude to -1..1. Rendering
    // that square coordinate domain to a 2:1 target restores the equirectangular
    // angular aspect (360 degrees by 180 degrees).
    let projection = Mat4::orthographic_rh(-1.1, 1.1, -1.1, 1.1, -1.0, 1.0);
    let uniforms = Uniforms::new(projection, Vec3::Z, Vec3::Z)
        .with_relief_scale(0.0)
        .with_slope_shading(false)
        .with_hemisphere_lighting(false)
        .with_map_mode(true)
        .with_rivers(false);
    renderer.render_to_view(
        &gpu.device,
        &gpu.queue,
        color_view,
        &uniforms,
        RenderScene {
            fill_pipeline: FillPipelineKind::Map,
            fill: IndexedDraw {
                vertex_buffer,
                index_buffer,
                index_count,
            },
            river_texture_bind_group: None,
            edges: None,
            arrows: None,
            pole_markers: None,
            rivers: None,
            gpu_particles: None,
        },
    );
}

/// Render the bounded living-surface semantic proof without turning its colors
/// into a product palette. Inputs, cover and exclusive fractions remain separate.
fn run_living_surface_preview(opts: &SweepOptions) {
    assert_eq!(
        opts.target_stage, 4,
        "living-surface-preview requires --stage 4"
    );
    let world = generate_tile_world(opts, &opts.base_erosion);
    let tess = world.active_tessellation();
    let hydrology = world.active_hydrology().expect("stage 4 hydrology");
    let semantic_started = std::time::Instant::now();
    let living_surface = LivingSurfaceSemantics::build(
        tess,
        world.active_temperature().expect("stage 4 temperature"),
        world.active_precipitation().expect("stage 4 precipitation"),
        hydrology,
    );
    let semantic_build_ms = semantic_started.elapsed().as_secs_f64() * 1_000.0;

    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|e| panic!("create {}: {e}", opts.out_dir.display()));
    let gpu = pollster::block_on(GpuContext::new_headless(opts.width, opts.height));
    let mut renderer = Renderer::new(&gpu, &Uniforms::new(Mat4::IDENTITY, Vec3::ZERO, Vec3::Y));
    let color_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("living_surface_preview_color"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&Default::default());
    let montage_w = opts.width * 4;
    let montage_h = opts.height * 2;
    let mut montage = vec![0; (montage_w * montage_h * 4) as usize];
    let mut files = Vec::new();

    for (index, layer) in LivingPreviewLayer::ALL.into_iter().enumerate() {
        let mut mesh = VoronoiMesh::from_voronoi_with_colors(&tess.voronoi, |cell| {
            living_preview_color(
                layer,
                &living_surface.cells[cell],
                hydrology.is_ocean(cell),
                hydrology.is_submerged(cell),
            )
        });
        // The legacy colored shader applies diffuse lighting unconditionally.
        // A constant normal keeps this semantic packet unlit and comparable.
        for vertex in &mut mesh.vertices {
            vertex.normal = Vec3::Z.to_array();
        }
        let vertices = create_vertex_buffer(
            &gpu.device,
            &mesh.vertices,
            "living_surface_preview_vertices",
        );
        let indices =
            create_index_buffer(&gpu.device, &mesh.indices, "living_surface_preview_indices");
        render_living_preview_map(
            &gpu,
            &mut renderer,
            &color_view,
            &vertices,
            &indices,
            mesh.indices.len() as u32,
        );
        let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
        let filename = format!("{:02}_{}.png", index + 1, layer.id());
        write_png(
            &opts.out_dir.join(&filename),
            &rgba,
            opts.width,
            opts.height,
        );
        blit_tile(
            &mut montage,
            montage_w,
            &rgba,
            opts.width,
            opts.height,
            (index % 4) as u32,
            (index / 4) as u32,
        );
        files.push(serde_json::json!({
            "id": layer.id(),
            "role": layer.role(),
            "filename": filename,
        }));
    }
    write_png(
        &opts.out_dir.join("montage.png"),
        &montage,
        montage_w,
        montage_h,
    );

    // Matched presentation discriminator: ordinary terrain at the default
    // authentic/cartographic scale, then the same fractional surface at all
    // three declared relief scales. Rivers stay off so they cannot make the
    // drainage-relative semantic signal appear stronger than it is.
    let views = build_views(&world, opts);
    let presentation_montage_w = opts.width * views.len() as u32;
    let presentation_montage_h = opts.height * 4;
    let mut presentation_montage =
        vec![0; (presentation_montage_w * presentation_montage_h * 4) as usize];
    let (_transparent_river_texture, transparent_river_bind_group) =
        transparent_river_binding(&gpu);
    let mut presentation_rows = Vec::new();

    let ordinary_mesh =
        living_presentation_mesh(&gpu.device, &world, |cell| cell_color_terrain(&world, cell));
    let mut ordinary_files = Vec::with_capacity(views.len());
    for (column, view) in views.iter().enumerate() {
        render_diagnostic(
            &gpu,
            &mut renderer,
            &color_view,
            &ordinary_mesh,
            &transparent_river_bind_group,
            view.view_proj,
            view.eye,
            ReliefPreset::Authentic.scale(),
            true,
        );
        let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
        let filename = format!("presentation_01_ordinary-authentic_{}.png", view.label);
        write_png(
            &opts.out_dir.join(&filename),
            &rgba,
            opts.width,
            opts.height,
        );
        blit_tile(
            &mut presentation_montage,
            presentation_montage_w,
            &rgba,
            opts.width,
            opts.height,
            column as u32,
            0,
        );
        ordinary_files.push(filename);
    }
    presentation_rows.push(serde_json::json!({
        "id": "ordinary-authentic",
        "role": "control: ordinary terrain colors without living-surface fractions",
        "semantic_blend": false,
        "relief_preset": ReliefPreset::Authentic.name(),
        "relief_scale": ReliefPreset::Authentic.scale(),
        "image_filenames": ordinary_files,
    }));
    let blended_mesh = living_presentation_mesh(&gpu.device, &world, |cell| {
        living_surface_blended_color(
            cell_color_terrain(&world, cell),
            living_surface.cells[cell].fractions,
            hydrology.is_submerged(cell),
        )
    });
    let relief_rows = [
        ("physical", ReliefPreset::Physical),
        ("authentic", ReliefPreset::Authentic),
        ("dramatic", ReliefPreset::Dramatic),
    ];
    for (row_offset, (id, preset)) in relief_rows.into_iter().enumerate() {
        let row = row_offset + 1;
        let mut filenames = Vec::with_capacity(views.len());
        for (column, view) in views.iter().enumerate() {
            render_diagnostic(
                &gpu,
                &mut renderer,
                &color_view,
                &blended_mesh,
                &transparent_river_bind_group,
                view.view_proj,
                view.eye,
                preset.scale(),
                true,
            );
            let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
            let filename = format!(
                "presentation_{:02}_living-{}_{}.png",
                row + 1,
                id,
                view.label
            );
            write_png(
                &opts.out_dir.join(&filename),
                &rgba,
                opts.width,
                opts.height,
            );
            blit_tile(
                &mut presentation_montage,
                presentation_montage_w,
                &rgba,
                opts.width,
                opts.height,
                column as u32,
                row as u32,
            );
            filenames.push(filename);
        }
        presentation_rows.push(serde_json::json!({
            "id": format!("living-{id}"),
            "role": "fractional living surface blended over the ordinary terrain substrate",
            "semantic_blend": true,
            "relief_preset": preset.name(),
            "relief_scale": preset.scale(),
            "image_filenames": filenames,
        }));
    }
    write_png(
        &opts.out_dir.join("presentation-montage.png"),
        &presentation_montage,
        presentation_montage_w,
        presentation_montage_h,
    );

    let areas = tess.cell_areas_ref();
    let (land_area, unresolved_area, max_closure_error, sums) =
        living_surface.cells.iter().enumerate().fold(
            (0.0f64, 0.0f64, 0.0f32, [0.0f64; 9]),
            |(area, unresolved, max_error, mut sums), (cell, state)| {
                if hydrology.is_submerged(cell) {
                    return (area, unresolved, max_error, sums);
                }
                let a = areas[cell] as f64;
                sums[0] += a * state.thermal_opportunity as f64;
                sums[1] += a * (1.0 - state.relative_water_limitation) as f64;
                sums[2] += a * state.drainage_saturation as f64;
                sums[3] += a * state.growth_opportunity as f64;
                sums[4] += a * state.vegetation_cover as f64;
                sums[5] += a * state.fractions.bare as f64;
                sums[6] += a * state.fractions.herbaceous as f64;
                sums[7] += a * state.fractions.woody as f64;
                sums[8] += a * state.fractions.wetland as f64;
                (
                    area + a,
                    unresolved
                        + if state.height_above_drainage_km.is_none() {
                            a
                        } else {
                            0.0
                        },
                    max_error.max((state.fractions.terrestrial_sum() - 1.0).abs()),
                    sums,
                )
            },
        );
    let means: Vec<f64> = sums.into_iter().map(|sum| sum / land_area).collect();
    let sidecar = serde_json::json!({
        "schema_version": 3,
        "purpose": "untuned evidence for the bounded equilibrium-physiognomy semantic kernel",
        "status": "implemented semantic proof; not retained world state, calibrated ecology, a biome map, or a product palette",
        "world_manifest": world.manifest(),
        "projection": "equirectangular; exact Voronoi cells; no relief displacement or semantic smoothing",
        "known_limitations": [
            "temperature and precipitation are seasonless normalized inputs",
            "planetary water-supply scale is fixed at 1.0 until one upstream control rebuilds precipitation and hydrology together",
            "HAND uses a 2000 km2 nominal geometric drainage reference with a four-local-cell adaptive-resolution floor",
            "drainage saturation uses a disclosed 30 m vertical decay and 0.35 subcell occupancy cap outside channel-reference cells",
            "channel-reference membership does not estimate channel or floodplain width, so reference cells claim no wetland occupancy",
            "terrain exposure is omitted until a scale-declared robust measure exists",
            "woody share has no seasonal, fire, disturbance, soil, or competition owner",
            "fractions are equilibrium opportunities, not persistent biomass or history"
        ],
        "planetary_water_supply_scale": living_surface.planetary_water_supply_scale,
        "drainage_reference_area_km2": living_surface.drainage_reference_area_km2,
        "minimum_drainage_reference_cells": living_surface.minimum_drainage_reference_cells,
        "semantic_build_ms": semantic_build_ms,
        "semantic_cell_bytes": std::mem::size_of::<hex3::world::LivingSurfaceCell>(),
        "area_weighted_land_means": {
            "thermal_opportunity": means[0],
            "climatic_water_availability": means[1],
            "drainage_saturation": means[2],
            "growth_opportunity": means[3],
            "vegetation_cover": means[4],
            "bare_fraction": means[5],
            "herbaceous_fraction": means[6],
            "woody_fraction": means[7],
            "wetland_fraction": means[8]
        },
        "land_fraction_without_drainage_reference": unresolved_area / land_area,
        "maximum_land_fraction_closure_error": max_closure_error,
        "layers": files,
        "montage_filename": "montage.png",
        "presentation": {
            "status": "diagnostic palette and matched-camera relief comparison; not a product palette",
            "color_contract": {
                "formula": "ordinary_terrain * bare + herbaceous_color * herbaceous + woody_color * woody + wetland_color * wetland; authoritative water keeps ordinary terrain color",
                "herbaceous_linear_rgb": [LIVING_HERBACEOUS_COLOR.x, LIVING_HERBACEOUS_COLOR.y, LIVING_HERBACEOUS_COLOR.z],
                "woody_linear_rgb": [LIVING_WOODY_COLOR.x, LIVING_WOODY_COLOR.y, LIVING_WOODY_COLOR.z],
                "wetland_linear_rgb": [LIVING_WETLAND_COLOR.x, LIVING_WETLAND_COLOR.y, LIVING_WETLAND_COLOR.z],
                "normalization": "none",
                "semantic_noise": "none",
                "rivers": "disabled to avoid confounding drainage-relative cover"
            },
            "sampling": "cell colors and elevation are averaged onto shared Voronoi vertices for the relief view; exact cell fractions remain in the flat scalar layers",
            "cameras": views.iter().map(|view| view.sidecar.clone()).collect::<Vec<_>>(),
            "rows": presentation_rows,
            "montage_filename": "presentation-montage.png"
        }
    });
    let file = std::fs::File::create(opts.out_dir.join("living-surface-preview.json"))
        .expect("create living-surface-preview.json");
    serde_json::to_writer_pretty(BufWriter::new(file), &sidecar)
        .expect("write living-surface-preview.json");
    println!("Done: living-surface preview -> {}", opts.out_dir.display());
}

fn terrain_slope(tess: &Tessellation, elevation: &[f32]) -> Vec<f32> {
    (0..tess.num_cells())
        .map(|i| {
            let p = tess.cell_center(i);
            tess.neighbors(i)
                .iter()
                .map(|&j| {
                    let arc = p.dot(tess.cell_center(j)).clamp(-1.0, 1.0).acos();
                    (elevation[j] - elevation[i]).abs() / arc.max(1e-7)
                })
                .fold(0.0, f32::max)
        })
        .collect()
}

/// Produce the one-world, matched-camera packet that discriminates where broad,
/// flat-topped mountain morphology enters the product pipeline.
fn run_range_ancestry(opts: &SweepOptions) {
    assert_eq!(opts.target_stage, 4, "range-ancestry requires --stage 4");
    assert_eq!(
        opts.targets.len(),
        3,
        "range-ancestry requires exactly three explicit --sweep-target dossier range cameras"
    );
    assert_eq!(
        selected_orogen_model(opts.orogen_model, &opts.base_erosion),
        OrogenModel::Legacy,
        "range-ancestry reconstructs the default legacy uplift source only"
    );

    let world = generate_tile_world(opts, &opts.base_erosion);
    let fine = world
        .fine
        .as_ref()
        .expect("range-ancestry requires fine stage 4");
    let final_surface = fine
        .eroded
        .as_ref()
        .expect("range-ancestry requires eroded surface");
    assert_eq!(
        fine.base.emergent_lambda, 0.0,
        "range-ancestry legacy uplift reconstruction requires emergent_lambda=0"
    );
    assert!(
        fine.base.fields.elevation_fields.legacy_uplift_source,
        "selected elevation fields do not authorize legacy repeated uplift"
    );
    assert_eq!(
        world.erosion_params.uplift_smooth_km, 0.0,
        "range-ancestry can exactly reconstruct only the default unsmoothed legacy uplift source"
    );

    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|e| panic!("create {}: {e}", opts.out_dir.display()));
    let gpu = pollster::block_on(GpuContext::new_headless(opts.width, opts.height));
    let mut renderer = Renderer::new(&gpu, &Uniforms::new(Mat4::IDENTITY, Vec3::ZERO, Vec3::Y));
    let color_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("range_ancestry_color"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&Default::default());
    // Reuse its transparent/disabled river binding; diagnostic meshes have no rivers.
    let product_buffers = generate_world_buffers(&gpu.device, &gpu.queue, &world);
    let views: Vec<CaptureView> = build_views(&world, opts).into_iter().skip(1).collect();

    let coarse = &world.tessellation;
    let coarse_elevation = &world.elevation.as_ref().expect("coarse elevation").values;
    let fine_tess = &fine.base.tessellation;
    let raw_eroded: Vec<f32> = (0..fine_tess.num_cells())
        .map(|i| final_surface.hydrology.pre_integration_elevation(i))
        .collect();
    let final_elevation = &final_surface.hydrology.elevation;
    let pre_integrated = &fine.pre.hydrology.elevation;
    let ef = &fine.base.fields.elevation_fields;
    let legacy_uplift: Vec<f32> = (0..fine_tess.num_cells())
        .map(|i| {
            if fine.base.base_elevation[i] < 0.0 {
                0.0
            } else {
                world.erosion_params.uplift_scale * (ef.tectonic_thickening[i] + ef.rift_delta[i])
            }
        })
        .collect();
    let slope = terrain_slope(fine_tess, final_elevation);
    let log_flow: Vec<f32> = final_surface
        .hydrology
        .flow_accumulation
        .iter()
        .map(|&v| v.max(0.0).ln_1p())
        .collect();
    let cuts: Vec<f32> = (0..fine_tess.num_cells())
        .map(|i| final_surface.hydrology.integration_cut_depth(i))
        .collect();

    struct Layer<'a> {
        id: &'static str,
        label: &'static str,
        tess: &'a Tessellation,
        elevation: &'a [f32],
        scalar: Option<&'a [f32]>,
        role: &'static str,
        source: &'static str,
        units: &'static str,
        relief: f32,
    }
    let layers = vec![
        Layer { id: "coarse-elevation", label: "coarse elevation", tess: coarse, elevation: coarse_elevation, scalar: None, role: "terrain", source: "World::elevation.values", units: "elevation units", relief: 0.04 },
        Layer { id: "fine-coarse-interpolant", label: "fine coarse interpolant", tess: fine_tess, elevation: &fine.base.coarse_base_elevation, scalar: None, role: "terrain", source: "FineBase::coarse_base_elevation", units: "elevation units", relief: 0.04 },
        Layer { id: "fine-base", label: "fine pre-erosion base", tess: fine_tess, elevation: &fine.base.base_elevation, scalar: None, role: "terrain", source: "FineBase::base_elevation", units: "elevation units", relief: 0.04 },
        Layer { id: "pre-erosion-integrated", label: "pre-erosion post-integration", tess: fine_tess, elevation: pre_integrated, scalar: None, role: "terrain", source: "FineWorld::pre.hydrology.elevation", units: "elevation units", relief: 0.04 },
        Layer { id: "raw-eroded", label: "raw eroded pre-integration", tess: fine_tess, elevation: &raw_eroded, scalar: None, role: "terrain", source: "FineWorld::eroded.hydrology.pre_integration_elevation", units: "elevation units", relief: 0.04 },
        Layer { id: "final", label: "final post-integration", tess: fine_tess, elevation: final_elevation, scalar: None, role: "terrain", source: "FineWorld::eroded.hydrology.elevation", units: "elevation units", relief: 0.04 },
        Layer { id: "tectonic-thickening", label: "tectonic thickening", tess: fine_tess, elevation: final_elevation, scalar: Some(&ef.tectonic_thickening), role: "scalar", source: "FineFields::elevation_fields.tectonic_thickening", units: "crust-thickness units", relief: 0.0 },
        Layer { id: "legacy-repeated-uplift", label: "exact default legacy repeated-uplift source", tess: fine_tess, elevation: final_elevation, scalar: Some(&legacy_uplift), role: "scalar", source: "uplift_scale * (tectonic_thickening + rift_delta), base-land gated; unsmoothed default", units: "thickness units per erosion step", relief: 0.0 },
        Layer { id: "final-slope", label: "final slope", tess: fine_tess, elevation: final_elevation, scalar: Some(&slope), role: "scalar", source: "max neighbor |delta elevation| / arc radians on final integrated terrain", units: "elevation units per radian", relief: 0.0 },
        Layer { id: "log-flow", label: "log flow / drainage", tess: fine_tess, elevation: final_elevation, scalar: Some(&log_flow), role: "scalar", source: "ln(1 + FineWorld::eroded.hydrology.flow_accumulation)", units: "log(1 + precipitation-weighted steradians)", relief: 0.0 },
        Layer { id: "integration-cut", label: "integration cut depth", tess: fine_tess, elevation: final_elevation, scalar: Some(&cuts), role: "scalar", source: "Hydrology::integration_cut_depth", units: "elevation units", relief: 0.0 },
    ];

    let montage_w = opts.width * views.len() as u32;
    let montage_h = opts.height * layers.len() as u32;
    let mut montage = vec![0; (montage_w * montage_h * 4) as usize];
    let mut records = Vec::with_capacity(layers.len());
    for (li, layer) in layers.iter().enumerate() {
        let scale = layer.scalar.map(robust_scale);
        let mesh = diagnostic_mesh(
            &gpu.device,
            layer.tess,
            layer.elevation,
            layer
                .scalar
                .zip(scale)
                .map(|(values, (lo, hi))| (values, lo, hi)),
        );
        let mut filenames = Vec::with_capacity(views.len());
        for (vi, view) in views.iter().enumerate() {
            render_diagnostic(
                &gpu,
                &mut renderer,
                &color_view,
                &mesh,
                &product_buffers.river_bind_group,
                view.view_proj,
                view.eye,
                layer.relief,
                layer.scalar.is_none(),
            );
            let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
            let filename = format!("{:02}_{}_{}.png", li + 1, layer.id, view.label);
            write_png(
                &opts.out_dir.join(&filename),
                &rgba,
                opts.width,
                opts.height,
            );
            blit_tile(
                &mut montage,
                montage_w,
                &rgba,
                opts.width,
                opts.height,
                vi as u32,
                li as u32,
            );
            filenames.push(filename);
        }
        records.push(RangeLayerRecord {
            index: li,
            id: layer.id,
            label: layer.label,
            topology: if layer.tess.num_cells() == coarse.num_cells() {
                "coarse"
            } else {
                "fine"
            },
            role: layer.role,
            source: layer.source,
            units: layer.units,
            relief_scale: layer.relief,
            robust_color_min: scale.map(|s| s.0),
            robust_color_max: scale.map(|s| s.1),
            image_filenames: filenames,
        });
        println!("[{}/{}] {}", li + 1, layers.len(), layer.label);
    }
    write_png(
        &opts.out_dir.join("montage.png"),
        &montage,
        montage_w,
        montage_h,
    );
    let sidecar = RangeAncestrySidecar {
        schema_version: 1,
        purpose: "matched-camera terrain ancestry for diagnosing broad, flat-topped mountain systems",
        coordinate_convention: "latitude=asin(y); longitude=atan2(z,x); degrees; positive longitude rotates +X toward +Z",
        color_sampling: "cell scalars are averaged onto shared Voronoi vertices for topology-consistent anti-aliasing; coarse and fine terrain use the same shared-vertex interpolation",
        world_manifest: world.manifest(),
        cameras: views.iter().map(|v| v.sidecar.clone()).collect(),
        layers: records,
        montage_filename: "montage.png",
    };
    let file = std::fs::File::create(opts.out_dir.join("range-ancestry.json"))
        .expect("create range-ancestry.json");
    serde_json::to_writer_pretty(BufWriter::new(file), &sidecar)
        .expect("write range-ancestry.json");
    println!("Done: range ancestry packet -> {}", opts.out_dir.display());
}

#[cfg(feature = "research-landscape")]
fn interpolate_roof_counterfactual(
    coarse: &Tessellation,
    fine: &Tessellation,
    coarse_cell: &[usize],
    values: &[f32],
) -> Vec<f32> {
    assert_eq!(coarse.num_cells(), values.len());
    assert_eq!(fine.num_cells(), coarse_cell.len());
    (0..fine.num_cells())
        .map(|cell| {
            let position = fine.cell_center(cell);
            let nearest = coarse_cell[cell];
            let mut weighted = 0.0f32;
            let mut total_weight = 0.0f32;
            for source in std::iter::once(nearest).chain(coarse.neighbors(nearest).iter().copied())
            {
                let distance = coarse
                    .cell_center(source)
                    .dot(position)
                    .clamp(-1.0, 1.0)
                    .acos();
                let weight = 1.0 / (distance * distance + 1.0e-8);
                weighted += values[source] * weight;
                total_weight += weight;
            }
            weighted / total_weight
        })
        .collect()
}

/// Render the two work-matched Legacy collision-compiler counterfactuals over
/// the retained product interpolation support. Both variants are static coarse
/// reconstructions: erosion runs once for the baseline/final context and is
/// never run against either counterfactual.
#[cfg(feature = "research-landscape")]
fn run_roof_compiler_counterfactual(opts: &SweepOptions) {
    assert_eq!(
        opts.target_stage, 4,
        "roof-compiler-counterfactual requires --stage 4"
    );
    assert_eq!(
        opts.targets.len(),
        2,
        "roof-compiler-counterfactual requires exactly two explicit --sweep-target dossier cameras"
    );
    assert_eq!(
        selected_orogen_model(opts.orogen_model, &opts.base_erosion),
        OrogenModel::Legacy,
        "roof-compiler-counterfactual requires the Legacy orogen model"
    );

    let world = generate_tile_world(opts, &opts.base_erosion);
    let features = world
        .features
        .as_ref()
        .expect("roof-compiler-counterfactual requires Legacy Stage-1 feature fields");
    let fine = world
        .fine
        .as_ref()
        .expect("roof-compiler-counterfactual requires fine Stage 4");
    let final_surface = fine
        .eroded
        .as_ref()
        .expect("roof-compiler-counterfactual requires final Stage-4 terrain");
    let coarse = &world.tessellation;
    let fine_tess = &fine.base.tessellation;
    let coarse_elevation = &world.elevation.as_ref().expect("coarse elevation").values;
    let trace = &features.legacy_collision_trace;
    let replace_collision = |replacement: &[f32]| -> Vec<f32> {
        coarse_elevation
            .iter()
            .zip(&features.collision)
            .zip(replacement)
            .map(|((&baseline, &collision), &counterfactual)| baseline - collision + counterfactual)
            .collect()
    };
    let reinterpolated_baseline = interpolate_roof_counterfactual(
        coarse,
        fine_tess,
        &fine.base.coarse_cell,
        coarse_elevation,
    );
    let baseline_reinterpolation_max_abs_error = reinterpolated_baseline
        .iter()
        .zip(&fine.base.coarse_base_elevation)
        .map(|(&reconstructed, &retained)| (reconstructed - retained).abs())
        .fold(0.0f32, f32::max);
    let nearest_source = interpolate_roof_counterfactual(
        coarse,
        fine_tess,
        &fine.base.coarse_cell,
        &replace_collision(&trace.nearest_source_matched_response),
    );
    let episode_mean = interpolate_roof_counterfactual(
        coarse,
        fine_tess,
        &fine.base.coarse_cell,
        &replace_collision(&trace.episode_mean_matched_response),
    );

    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|error| panic!("create {}: {error}", opts.out_dir.display()));
    let gpu = pollster::block_on(GpuContext::new_headless(opts.width, opts.height));
    let mut renderer = Renderer::new(&gpu, &Uniforms::new(Mat4::IDENTITY, Vec3::ZERO, Vec3::Y));
    let color_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("roof_compiler_counterfactual_color"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&Default::default());
    let product_buffers = generate_world_buffers(&gpu.device, &gpu.queue, &world);
    let views: Vec<CaptureView> = build_views(&world, opts).into_iter().skip(1).collect();
    let final_elevation = &final_surface.hydrology.elevation;

    struct Layer<'a> {
        id: &'static str,
        label: &'static str,
        elevation: &'a [f32],
        source: &'static str,
    }
    let layers = [
        Layer {
            id: "baseline-fine-coarse-interpolant",
            label: "baseline fine coarse interpolant",
            elevation: &fine.base.coarse_base_elevation,
            source: "FineBase::coarse_base_elevation retained from the product interpolation",
        },
        Layer {
            id: "nearest-source-work-matched",
            label: "nearest-source work-matched compiler surface",
            elevation: &nearest_source,
            source: "interpolate(World::elevation.values - FeatureFields::collision + LegacyCollisionTrace::nearest_source_matched_response)",
        },
        Layer {
            id: "episode-mean-null-work-matched",
            label: "episode-mean-null work-matched compiler surface",
            elevation: &episode_mean,
            source: "interpolate(World::elevation.values - FeatureFields::collision + LegacyCollisionTrace::episode_mean_matched_response)",
        },
        Layer {
            id: "final-terrain-context",
            label: "final terrain for context",
            elevation: final_elevation,
            source: "FineWorld::eroded.hydrology.elevation from the single baseline Stage-4 run",
        },
    ];

    let montage_width = opts.width * views.len() as u32;
    let montage_height = opts.height * layers.len() as u32;
    let mut montage = vec![0; (montage_width * montage_height * 4) as usize];
    let mut records = Vec::with_capacity(layers.len());
    for (layer_index, layer) in layers.iter().enumerate() {
        let mesh = diagnostic_mesh(&gpu.device, fine_tess, layer.elevation, None);
        let mut filenames = Vec::with_capacity(views.len());
        for (view_index, view) in views.iter().enumerate() {
            render_diagnostic(
                &gpu,
                &mut renderer,
                &color_view,
                &mesh,
                &product_buffers.river_bind_group,
                view.view_proj,
                view.eye,
                0.04,
                true,
            );
            let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
            let filename = format!("{:02}_{}_{}.png", layer_index + 1, layer.id, view.label);
            write_png(
                &opts.out_dir.join(&filename),
                &rgba,
                opts.width,
                opts.height,
            );
            blit_tile(
                &mut montage,
                montage_width,
                &rgba,
                opts.width,
                opts.height,
                view_index as u32,
                layer_index as u32,
            );
            filenames.push(filename);
        }
        records.push(RangeLayerRecord {
            index: layer_index,
            id: layer.id,
            label: layer.label,
            topology: "fine",
            role: "terrain",
            source: layer.source,
            units: "elevation units",
            relief_scale: 0.04,
            robust_color_min: None,
            robust_color_max: None,
            image_filenames: filenames,
        });
        println!("[{}/{}] {}", layer_index + 1, layers.len(), layer.label);
    }
    write_png(
        &opts.out_dir.join("montage.png"),
        &montage,
        montage_width,
        montage_height,
    );

    let sidecar = RoofCompilerCounterfactualSidecar {
        schema_version: 1,
        purpose: "matched-camera terrain comparison of the frozen Legacy collision compiler and two work-matched counterfactual compilers",
        status: "research-only causal discriminator; no product model, erosion variant, calibration, or tuning",
        coordinate_convention: "latitude=asin(y); longitude=atan2(z,x); degrees; positive longitude rotates +X toward +Z",
        color_sampling: "all terrain rows use the same fine Voronoi topology, shared-vertex elevation averaging, material color, lighting, relief scale, and dossier cameras",
        world_manifest: world.manifest(),
        counterfactuals: RoofCompilerCounterfactualContract {
            baseline_coarse_elevation: "World::elevation.values",
            baseline_collision_response: "World::features.collision (FeatureFields::collision)",
            reconstruction_formula: "counterfactual coarse elevation = baseline coarse elevation - FeatureFields::collision + research trace matched response",
            nearest_source_response: "LegacyCollisionTrace::nearest_source_matched_response",
            episode_mean_response: "LegacyCollisionTrace::episode_mean_matched_response",
            interpolation_support: "mapped nearest coarse cell followed by that coarse cell's neighbors",
            interpolation_weight: "1 / (angular_distance_radians^2 + epsilon)",
            interpolation_epsilon: 1.0e-8,
            baseline_reinterpolation_max_abs_error,
            work_matching_measure: "area-integrated positive collision response on the coarse tessellation",
            nearest_source_work_scale: trace.nearest_source_work_scale,
            episode_mean_work_scale: trace.episode_mean_work_scale,
            erosion_policy: "erosion is run once for baseline Stage 4 only; neither compiler counterfactual is eroded",
        },
        cameras: views.iter().map(|view| view.sidecar.clone()).collect(),
        layers: records,
        montage_filename: "montage.png",
    };
    let file = std::fs::File::create(opts.out_dir.join("roof-compiler-counterfactual.json"))
        .expect("create roof-compiler-counterfactual.json");
    serde_json::to_writer_pretty(BufWriter::new(file), &sidecar)
        .expect("write roof-compiler-counterfactual.json");
    println!(
        "Done: roof compiler counterfactual packet -> {}",
        opts.out_dir.display()
    );
}

#[derive(Clone, Debug, Serialize, PartialEq)]
struct SiteComparisonRecord {
    exact_retained_anchor_count: usize,
    exact_retained_anchor_fraction: f32,
    mean_nearest_baseline_site_distance_km: Option<f32>,
    median_nearest_baseline_site_distance_km: Option<f32>,
    maximum_nearest_baseline_site_distance_km: Option<f32>,
}

fn compare_site_positions(baseline: &[Vec3], variant: &[Vec3]) -> SiteComparisonRecord {
    let exact_retained_anchor_count = variant
        .iter()
        .filter(|position| baseline.iter().any(|base| base == *position))
        .count();
    let mut nearest: Vec<f32> = variant
        .iter()
        .filter_map(|position| {
            baseline
                .iter()
                .map(|base| {
                    if position == base {
                        0.0
                    } else {
                        position.dot(*base).clamp(-1.0, 1.0).acos() * PLANET_RADIUS_KM
                    }
                })
                .min_by(|a, b| a.total_cmp(b))
        })
        .collect();
    nearest.sort_by(f32::total_cmp);
    let (mean, median, maximum) = if nearest.is_empty() {
        (None, None, None)
    } else {
        let mean = nearest.iter().sum::<f32>() / nearest.len() as f32;
        let middle = nearest.len() / 2;
        let median = if nearest.len().is_multiple_of(2) {
            (nearest[middle - 1] + nearest[middle]) * 0.5
        } else {
            nearest[middle]
        };
        (Some(mean), Some(median), nearest.last().copied())
    };
    SiteComparisonRecord {
        exact_retained_anchor_count,
        exact_retained_anchor_fraction: if variant.is_empty() {
            0.0
        } else {
            exact_retained_anchor_count as f32 / variant.len() as f32
        },
        mean_nearest_baseline_site_distance_km: mean,
        median_nearest_baseline_site_distance_km: median,
        maximum_nearest_baseline_site_distance_km: maximum,
    }
}

fn site_positions(tess: &Tessellation, selection: &AggregateSiteSelection) -> Vec<Vec3> {
    selection
        .sites
        .iter()
        .map(|site| tess.cell_center(site.anchor_cell))
        .collect()
}

fn baseline_site_probe_config() -> SiteSelectionConfig {
    SiteSelectionConfig {
        site_count: 20,
        candidate_pool_size: 512,
        maximum_total_catchment_cell_visits: 2_000_000,
        minimum_site_spacing_km: 900.0,
        candidate_spacing_km: 200.0,
        catchment_budget_generalized_km: 450.0,
        freshwater_access_limit_generalized_km: 120.0,
        minimum_local_living_opportunity: 0.08,
        maximum_local_trimmed_mean_grade: 0.15,
        minimum_effective_catchment_area_km2: 15_000.0,
        coast_access_scale_generalized_km: 500.0,
        coast_bonus: 0.20,
    }
}

fn diagnostic_coarse_support_site_probe_config() -> SiteSelectionConfig {
    SiteSelectionConfig {
        candidate_pool_size: 160,
        ..baseline_site_probe_config()
    }
}

fn tight_site_probe_config() -> SiteSelectionConfig {
    SiteSelectionConfig {
        minimum_site_spacing_km: 1_100.0,
        candidate_spacing_km: 250.0,
        catchment_budget_generalized_km: 350.0,
        freshwater_access_limit_generalized_km: 90.0,
        minimum_local_living_opportunity: 0.12,
        maximum_local_trimmed_mean_grade: 0.12,
        minimum_effective_catchment_area_km2: 20_000.0,
        coast_access_scale_generalized_km: 400.0,
        coast_bonus: 0.15,
        ..baseline_site_probe_config()
    }
}

fn loose_site_probe_config() -> SiteSelectionConfig {
    SiteSelectionConfig {
        minimum_site_spacing_km: 700.0,
        candidate_spacing_km: 150.0,
        catchment_budget_generalized_km: 600.0,
        freshwater_access_limit_generalized_km: 160.0,
        minimum_local_living_opportunity: 0.04,
        maximum_local_trimmed_mean_grade: 0.20,
        minimum_effective_catchment_area_km2: 10_000.0,
        coast_access_scale_generalized_km: 650.0,
        coast_bonus: 0.25,
        ..baseline_site_probe_config()
    }
}

fn route_probe_config() -> RouteNetworkConfig {
    RouteNetworkConfig {
        nearest_neighbors_per_site: 4,
        maximum_candidate_pair_count: 96,
        maximum_total_search_cell_visits: 10_000_000,
        maximum_extra_links: 3,
        minimum_extra_link_detour_ratio: 1.35,
    }
}

#[derive(Clone, Debug, Serialize)]
struct RouteComparisonRecord {
    identical_site_anchor_input: bool,
    shared_selected_endpoint_pair_count: usize,
    baseline_only_selected_endpoint_pair_count: usize,
    zero_grade_only_selected_endpoint_pair_count: usize,
    selected_endpoint_pair_jaccard: f32,
    shared_selected_cell_edge_count: usize,
    selected_cell_edge_union_count: usize,
    selected_cell_edge_jaccard: f32,
    exact_candidate_path_count: usize,
    selected_candidate_path_count: usize,
    exact_selected_candidate_path_count: usize,
    mean_candidate_path_hausdorff_km: f32,
    median_candidate_path_hausdorff_km: f32,
    maximum_candidate_path_hausdorff_km: f32,
    maximum_selected_path_hausdorff_km: f32,
    most_divergent_candidate_route_id: Option<usize>,
    most_divergent_candidate_endpoint_site_ids: Option<[usize; 2]>,
    most_divergent_selected_candidate_route_id: Option<usize>,
    most_divergent_selected_endpoint_site_ids: Option<[usize; 2]>,
    most_divergent_selected_target_cell: Option<usize>,
    baseline_selected_physical_length_sum_km: f32,
    zero_grade_selected_physical_length_sum_km: f32,
    baseline_selected_ascent_sum_km: f32,
    zero_grade_selected_ascent_sum_km: f32,
}

#[derive(Debug, Serialize)]
struct RouteProbeVariant {
    id: &'static str,
    role: &'static str,
    line_color_linear_rgb: [f32; 3],
    build_ms: f64,
    image_filenames: Vec<String>,
    network: AggregateRouteNetwork,
    #[serde(skip)]
    line_color: Vec3,
}

#[derive(Debug, Serialize)]
struct RouteLowerCorridorRenderRecord {
    filename: String,
    camera: ViewRecord,
    physical_branch_color_linear_rgb: [f32; 3],
    distance_null_branch_color_linear_rgb: [f32; 3],
    divergence_endpoint_color_linear_rgb: [f32; 3],
    physical_crest_color_linear_rgb: [f32; 3],
    distance_null_crest_color_linear_rgb: [f32; 3],
}

#[derive(Debug, Serialize)]
struct RouteLowerCorridorProbeRecord {
    status: &'static str,
    physical_selected_candidate_count: usize,
    explained_candidate_count: usize,
    selection_rule: &'static str,
    strongest: Option<RouteLowerCorridorEvidence>,
    omissions: Vec<RouteLowerCorridorOmission>,
    render: Option<RouteLowerCorridorRenderRecord>,
    claim: &'static str,
    limitation: &'static str,
}

fn lower_corridor_probe_record(
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    physical: &AggregateRouteNetwork,
    distance_null: &AggregateRouteNetwork,
) -> Result<RouteLowerCorridorProbeRecord, &'static str> {
    let mut strongest: Option<RouteLowerCorridorEvidence> = None;
    let mut omissions = Vec::new();
    let mut physical_selected_candidate_count = 0usize;
    let mut explained_candidate_count = 0usize;
    for candidate_route_id in 0..physical.candidate_routes.len() {
        if physical.candidate_routes[candidate_route_id]
            .selection_role
            .is_none()
        {
            continue;
        }
        physical_selected_candidate_count += 1;
        match assess_route_lower_corridor(
            tess,
            hydrology,
            physical,
            distance_null,
            candidate_route_id,
        )? {
            RouteLowerCorridorAssessment::Explained { evidence } => {
                explained_candidate_count += 1;
                let replace = strongest.as_ref().is_none_or(|best| {
                    evidence
                        .generalized_cost_saved_km
                        .total_cmp(&best.generalized_cost_saved_km)
                        .then_with(|| {
                            evidence
                                .maximum_elevation_saved_km
                                .total_cmp(&best.maximum_elevation_saved_km)
                        })
                        .then_with(|| best.endpoint_site_ids.cmp(&evidence.endpoint_site_ids))
                        .then_with(|| {
                            best.divergence_endpoint_cells
                                .cmp(&evidence.divergence_endpoint_cells)
                        })
                        .is_gt()
                });
                if replace {
                    strongest = Some(*evidence);
                }
            }
            RouteLowerCorridorAssessment::Omitted { omission } => omissions.push(omission),
        }
    }
    Ok(RouteLowerCorridorProbeRecord {
        status: if physical_selected_candidate_count == 0 {
            "not-assessed"
        } else if strongest.is_some() {
            "explained"
        } else {
            "omitted"
        },
        physical_selected_candidate_count,
        explained_candidate_count,
        selection_rule: "among physically selected endpoint pairs, compare the already-retained same-endpoint distance-null path and choose the qualifying elementary divergence bubble with greatest physical generalized-cost saving, then greatest maximum-elevation saving, then lower endpoint and boundary cell identities",
        strongest,
        omissions,
        render: None,
        claim: "the physical route uses a longer but cheaper and lower terrain corridor than the matched distance-minimizing branch between the same split and rejoin cells",
        limitation: "this counterfactual establishes neither a geomorphic gap, pass, ridge or barrier nor an enumerated second-best physical route, road, construction choice or chokepoint",
    })
}

fn canonical_path_edges(path: &[usize]) -> BTreeSet<(usize, usize)> {
    path.windows(2)
        .map(|pair| (pair[0].min(pair[1]), pair[0].max(pair[1])))
        .collect()
}

fn selected_endpoint_pairs(network: &AggregateRouteNetwork) -> BTreeSet<(usize, usize)> {
    network
        .selected_route_ids
        .iter()
        .map(|&id| {
            let route = &network.candidate_routes[id];
            (route.from_site_id, route.to_site_id)
        })
        .collect()
}

fn selected_cell_edges(network: &AggregateRouteNetwork) -> BTreeSet<(usize, usize)> {
    network
        .selected_route_ids
        .iter()
        .flat_map(|&id| canonical_path_edges(&network.candidate_routes[id].ordered_cells))
        .collect()
}

fn path_hausdorff_target(
    tess: &Tessellation,
    baseline: &[usize],
    variant: &[usize],
) -> (f32, usize) {
    fn directed(tess: &Tessellation, from: &[usize], to: &[usize]) -> (f32, usize) {
        let mut best = (0.0f32, from[0]);
        for &cell in from {
            let position = tess.cell_center(cell);
            let nearest = to
                .iter()
                .map(|&other| (position - tess.cell_center(other)).length_squared())
                .min_by(f32::total_cmp)
                .unwrap_or(0.0);
            if nearest > best.0 || (nearest.to_bits() == best.0.to_bits() && cell < best.1) {
                best = (nearest, cell);
            }
        }
        best
    }
    let forward = directed(tess, baseline, variant);
    let reverse = directed(tess, variant, baseline);
    let (chord_squared, cell) = if reverse.0 > forward.0 {
        reverse
    } else {
        forward
    };
    let chord = chord_squared.sqrt().clamp(0.0, 2.0);
    (2.0 * PLANET_RADIUS_KM * (0.5 * chord).asin(), cell)
}

fn compare_route_networks(
    tess: &Tessellation,
    baseline: &AggregateRouteNetwork,
    zero_grade: &AggregateRouteNetwork,
) -> RouteComparisonRecord {
    assert_eq!(
        baseline.candidate_routes.len(),
        zero_grade.candidate_routes.len(),
        "matched route counterfactuals require identical candidate pairs"
    );
    let baseline_pairs = selected_endpoint_pairs(baseline);
    let zero_pairs = selected_endpoint_pairs(zero_grade);
    let shared_pair_count = baseline_pairs.intersection(&zero_pairs).count();
    let pair_union_count = baseline_pairs.union(&zero_pairs).count();
    let baseline_edges = selected_cell_edges(baseline);
    let zero_edges = selected_cell_edges(zero_grade);
    let shared_edge_count = baseline_edges.intersection(&zero_edges).count();
    let edge_union_count = baseline_edges.union(&zero_edges).count();

    let mut distances = Vec::with_capacity(baseline.candidate_routes.len());
    let mut exact_candidate_path_count = 0usize;
    let mut selected_candidate_path_count = 0usize;
    let mut exact_selected_candidate_path_count = 0usize;
    let mut most_divergent_candidate: Option<(usize, f32, usize)> = None;
    let mut most_divergent_selected: Option<(usize, f32, usize)> = None;
    for (index, (physical, flat)) in baseline
        .candidate_routes
        .iter()
        .zip(&zero_grade.candidate_routes)
        .enumerate()
    {
        assert_eq!(
            (physical.from_site_id, physical.to_site_id),
            (flat.from_site_id, flat.to_site_id),
            "matched route counterfactuals require identical endpoint order"
        );
        let exact = physical.ordered_cells == flat.ordered_cells
            || physical
                .ordered_cells
                .iter()
                .eq(flat.ordered_cells.iter().rev());
        exact_candidate_path_count += usize::from(exact);
        let selected = physical.selection_role.is_some() || flat.selection_role.is_some();
        selected_candidate_path_count += usize::from(selected);
        exact_selected_candidate_path_count += usize::from(selected && exact);
        let (distance, target_cell) =
            path_hausdorff_target(tess, &physical.ordered_cells, &flat.ordered_cells);
        distances.push(distance);
        let replace_candidate =
            most_divergent_candidate.is_none_or(|(best_index, best_distance, _)| {
                distance > best_distance
                    || (distance.to_bits() == best_distance.to_bits()
                        && (physical.from_site_id, physical.to_site_id)
                            < (
                                baseline.candidate_routes[best_index].from_site_id,
                                baseline.candidate_routes[best_index].to_site_id,
                            ))
            });
        if replace_candidate {
            most_divergent_candidate = Some((index, distance, target_cell));
        }
        if selected && distance > 0.0 {
            let replace = most_divergent_selected.is_none_or(|(best_index, best_distance, _)| {
                distance > best_distance
                    || (distance.to_bits() == best_distance.to_bits()
                        && (physical.from_site_id, physical.to_site_id)
                            < (
                                baseline.candidate_routes[best_index].from_site_id,
                                baseline.candidate_routes[best_index].to_site_id,
                            ))
            });
            if replace {
                most_divergent_selected = Some((index, distance, target_cell));
            }
        }
    }
    distances.sort_by(f32::total_cmp);
    let mean = if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    };
    let median = if distances.is_empty() {
        0.0
    } else if distances.len().is_multiple_of(2) {
        (distances[distances.len() / 2 - 1] + distances[distances.len() / 2]) * 0.5
    } else {
        distances[distances.len() / 2]
    };
    let (most_candidate_id, maximum_candidate) = most_divergent_candidate
        .map(|(id, distance, _)| (Some(id), distance))
        .unwrap_or((None, 0.0));
    let sum_selected = |network: &AggregateRouteNetwork,
                        field: fn(usize, &AggregateRouteNetwork) -> f32| {
        network
            .selected_route_ids
            .iter()
            .map(|&id| field(id, network))
            .sum()
    };
    let (most_selected_id, maximum_selected, most_selected_cell) = most_divergent_selected
        .map(|(id, distance, cell)| (Some(id), distance, Some(cell)))
        .unwrap_or((None, 0.0, None));
    RouteComparisonRecord {
        identical_site_anchor_input: baseline.site_anchor_cells == zero_grade.site_anchor_cells,
        shared_selected_endpoint_pair_count: shared_pair_count,
        baseline_only_selected_endpoint_pair_count: baseline_pairs.len() - shared_pair_count,
        zero_grade_only_selected_endpoint_pair_count: zero_pairs.len() - shared_pair_count,
        selected_endpoint_pair_jaccard: if pair_union_count == 0 {
            1.0
        } else {
            shared_pair_count as f32 / pair_union_count as f32
        },
        shared_selected_cell_edge_count: shared_edge_count,
        selected_cell_edge_union_count: edge_union_count,
        selected_cell_edge_jaccard: if edge_union_count == 0 {
            1.0
        } else {
            shared_edge_count as f32 / edge_union_count as f32
        },
        exact_candidate_path_count,
        selected_candidate_path_count,
        exact_selected_candidate_path_count,
        mean_candidate_path_hausdorff_km: mean,
        median_candidate_path_hausdorff_km: median,
        maximum_candidate_path_hausdorff_km: maximum_candidate,
        maximum_selected_path_hausdorff_km: maximum_selected,
        most_divergent_candidate_route_id: most_candidate_id,
        most_divergent_candidate_endpoint_site_ids: most_candidate_id.map(|id| {
            [
                baseline.candidate_routes[id].from_site_id,
                baseline.candidate_routes[id].to_site_id,
            ]
        }),
        most_divergent_selected_candidate_route_id: most_selected_id,
        most_divergent_selected_endpoint_site_ids: most_selected_id.map(|id| {
            [
                baseline.candidate_routes[id].from_site_id,
                baseline.candidate_routes[id].to_site_id,
            ]
        }),
        most_divergent_selected_target_cell: most_selected_cell,
        baseline_selected_physical_length_sum_km: sum_selected(baseline, |id, network| {
            network.candidate_routes[id].physical_length_km
        }),
        zero_grade_selected_physical_length_sum_km: sum_selected(zero_grade, |id, network| {
            network.candidate_routes[id].physical_length_km
        }),
        baseline_selected_ascent_sum_km: sum_selected(baseline, |id, network| {
            network.candidate_routes[id].ascent_km_from_from_site
        }),
        zero_grade_selected_ascent_sum_km: sum_selected(zero_grade, |id, network| {
            network.candidate_routes[id].ascent_km_from_from_site
        }),
    }
}

#[derive(Debug, Serialize)]
struct ConsequentialVariantRecord {
    id: &'static str,
    role: &'static str,
    changed_prior_or_factor: &'static str,
    marker_color_linear_rgb: [f32; 3],
    selection_build_ms: f64,
    comparison_to_baseline: SiteComparisonRecord,
    image_filenames: Vec<String>,
    selection: AggregateSiteSelection,
}

struct ConsequentialVariant {
    record: ConsequentialVariantRecord,
    marker_color: Vec3,
}

#[allow(clippy::too_many_arguments)]
fn select_probe_variant(
    id: &'static str,
    role: &'static str,
    changed_prior_or_factor: &'static str,
    marker_color: Vec3,
    components: &ConsequentialGeographyComponents,
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    config: SiteSelectionConfig,
    baseline_positions: Option<&[Vec3]>,
) -> ConsequentialVariant {
    let started = std::time::Instant::now();
    let selection = components
        .select_sites(tess, hydrology, config)
        .unwrap_or_else(|error| panic!("consequential-geography {id} selection: {error}"));
    let selection_build_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let positions = site_positions(tess, &selection);
    let comparison_to_baseline =
        compare_site_positions(baseline_positions.unwrap_or(&positions), &positions);
    ConsequentialVariant {
        record: ConsequentialVariantRecord {
            id,
            role,
            changed_prior_or_factor,
            marker_color_linear_rgb: marker_color.to_array(),
            selection_build_ms,
            comparison_to_baseline,
            image_filenames: Vec::new(),
            selection,
        },
        marker_color,
    }
}

fn orbit_probe_view(
    id: &str,
    kind: &'static str,
    yaw_deg: f32,
    pitch_deg: f32,
    distance: f32,
    aspect: f32,
) -> CaptureView {
    let mut camera = OrbitCamera::new();
    camera.yaw = yaw_deg.to_radians();
    camera.pitch = pitch_deg.to_radians();
    camera.distance = distance;
    camera.aspect = aspect;
    let eye = camera.eye_position();
    CaptureView {
        view_proj: camera.view_projection(),
        eye,
        label: id.to_string(),
        sidecar: ViewRecord {
            id: id.to_string(),
            kind,
            target: None,
            camera: CameraRecord {
                eye_xyz: vec3_array(eye),
                aim_xyz: [0.0; 3],
                up_xyz: [0.0, 1.0, 0.0],
                vertical_fov_deg: 45.0,
                aspect,
                near: 0.01,
                far: 10.0,
                target_altitude: None,
                orbit_yaw_deg: Some(yaw_deg),
                orbit_pitch_deg: Some(pitch_deg),
                orbit_distance: Some(distance),
            },
        },
    }
}

fn consequential_probe_views(
    opts: &SweepOptions,
    tess: &Tessellation,
    baseline: &AggregateSiteSelection,
    route_divergence_target_cell: Option<usize>,
) -> Vec<CaptureView> {
    let aspect = opts.width as f32 / opts.height as f32;
    let mut views = vec![
        orbit_probe_view(
            "globe-a",
            "matched-globe",
            opts.yaw_deg,
            opts.pitch_deg,
            opts.distance,
            aspect,
        ),
        orbit_probe_view(
            "globe-b",
            "matched-opposite-globe",
            opts.yaw_deg + 180.0,
            -opts.pitch_deg,
            opts.distance,
            aspect,
        ),
    ];
    let target_cell = route_divergence_target_cell
        .or_else(|| baseline.sites.first().map(|site| site.anchor_cell))
        .or_else(|| (0..tess.num_cells()).next())
        .expect("consequential-geography packet requires a non-empty tessellation");
    let (regional_id, regional_kind) = if route_divergence_target_cell.is_some() {
        ("regional-route-divergence", "derived-route-divergence")
    } else {
        ("regional-baseline-site-1", "derived-baseline-site")
    };
    let mut regional = derived_capture_view(
        regional_id,
        tess.cell_center(target_cell),
        aspect,
        opts.zoom_alt,
    );
    regional.sidecar.kind = regional_kind;
    views.push(regional);
    views
}

fn site_marker_vertices(
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    selection: &AggregateSiteSelection,
    color: Vec3,
) -> Vec<SurfaceVertex> {
    const SEGMENTS: usize = 16;
    const RING_RADIUS: f32 = 0.025;
    const CROSS_RADIUS: f32 = 0.036;
    let mut vertices = Vec::with_capacity(selection.sites.len() * (SEGMENTS * 2 + 4));
    for site in &selection.sites {
        let normal = tess.cell_center(site.anchor_cell).normalize();
        let reference = if normal.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let east = normal.cross(reference).normalize();
        let north = east.cross(normal).normalize();
        let elevation = hydrology.elevation[site.anchor_cell];
        let point = |angle: f32, radius: f32| {
            let tangent = east * angle.cos() + north * angle.sin();
            (normal * radius.cos() + tangent * radius.sin()).normalize()
        };
        for segment in 0..SEGMENTS {
            let a = segment as f32 * std::f32::consts::TAU / SEGMENTS as f32;
            let b = (segment + 1) as f32 * std::f32::consts::TAU / SEGMENTS as f32;
            vertices.push(SurfaceVertex::new(
                point(a, RING_RADIUS),
                elevation,
                color,
                1.0,
            ));
            vertices.push(SurfaceVertex::new(
                point(b, RING_RADIUS),
                elevation,
                color,
                1.0,
            ));
        }
        for angle in [0.0, std::f32::consts::FRAC_PI_2] {
            vertices.push(SurfaceVertex::new(
                point(angle, CROSS_RADIUS),
                elevation,
                color,
                1.0,
            ));
            vertices.push(SurfaceVertex::new(
                point(angle + std::f32::consts::PI, CROSS_RADIUS),
                elevation,
                color,
                1.0,
            ));
        }
    }
    vertices
}

fn route_overlay_vertices(
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    selection: &AggregateSiteSelection,
    network: &AggregateRouteNetwork,
    route_color: Vec3,
    marker_color: Vec3,
) -> Vec<SurfaceVertex> {
    let route_vertex_count = network
        .selected_route_ids
        .iter()
        .map(|&id| {
            network.candidate_routes[id]
                .ordered_cells
                .len()
                .saturating_sub(1)
                * 2
        })
        .sum::<usize>();
    let mut vertices = Vec::with_capacity(route_vertex_count + selection.sites.len() * 36);
    for &route_id in &network.selected_route_ids {
        for edge in network.candidate_routes[route_id].ordered_cells.windows(2) {
            for &cell in edge {
                vertices.push(SurfaceVertex::new(
                    tess.cell_center(cell),
                    hydrology.elevation[cell],
                    route_color,
                    1.0,
                ));
            }
        }
    }
    vertices.extend(site_marker_vertices(
        tess,
        hydrology,
        selection,
        marker_color,
    ));
    vertices
}

fn append_route_path_vertices(
    vertices: &mut Vec<SurfaceVertex>,
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    path: &[usize],
    color: Vec3,
) {
    for edge in path.windows(2) {
        for &cell in edge {
            vertices.push(SurfaceVertex::new(
                tess.cell_center(cell),
                hydrology.elevation[cell],
                color,
                1.0,
            ));
        }
    }
}

fn route_marker_frame(tess: &Tessellation, cell: usize) -> (Vec3, Vec3, Vec3) {
    let normal = tess.cell_center(cell).normalize();
    let reference = if normal.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let east = normal.cross(reference).normalize();
    let north = east.cross(normal).normalize();
    (normal, east, north)
}

fn append_route_ring_marker(
    vertices: &mut Vec<SurfaceVertex>,
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    cell: usize,
    color: Vec3,
) {
    const SEGMENTS: usize = 16;
    const RADIUS: f32 = 0.008;
    let (normal, east, north) = route_marker_frame(tess, cell);
    let point = |angle: f32| {
        let tangent = east * angle.cos() + north * angle.sin();
        (normal * RADIUS.cos() + tangent * RADIUS.sin()).normalize()
    };
    for segment in 0..SEGMENTS {
        let a = segment as f32 * std::f32::consts::TAU / SEGMENTS as f32;
        let b = (segment + 1) as f32 * std::f32::consts::TAU / SEGMENTS as f32;
        vertices.push(SurfaceVertex::new(
            point(a),
            hydrology.elevation[cell],
            color,
            1.0,
        ));
        vertices.push(SurfaceVertex::new(
            point(b),
            hydrology.elevation[cell],
            color,
            1.0,
        ));
    }
}

fn append_route_cross_marker(
    vertices: &mut Vec<SurfaceVertex>,
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    cell: usize,
    color: Vec3,
) {
    const RADIUS: f32 = 0.012;
    let (normal, east, north) = route_marker_frame(tess, cell);
    let point = |angle: f32| {
        let tangent = east * angle.cos() + north * angle.sin();
        (normal * RADIUS.cos() + tangent * RADIUS.sin()).normalize()
    };
    for angle in [0.0, std::f32::consts::FRAC_PI_2] {
        vertices.push(SurfaceVertex::new(
            point(angle),
            hydrology.elevation[cell],
            color,
            1.0,
        ));
        vertices.push(SurfaceVertex::new(
            point(angle + std::f32::consts::PI),
            hydrology.elevation[cell],
            color,
            1.0,
        ));
    }
}

fn lower_corridor_overlay_vertices(
    tess: &Tessellation,
    hydrology: &hex3::world::Hydrology,
    physical_network: &AggregateRouteNetwork,
    distance_null_network: &AggregateRouteNetwork,
    evidence: &RouteLowerCorridorEvidence,
) -> Vec<SurfaceVertex> {
    let physical = &physical_network.candidate_routes[evidence.candidate_route_id];
    let distance_null = &distance_null_network.candidate_routes[evidence.candidate_route_id];
    let physical_path = &physical.ordered_cells
        [evidence.physical_segment_cell_indices[0]..=evidence.physical_segment_cell_indices[1]];
    let distance_null_path = &distance_null.ordered_cells[evidence
        .distance_null_segment_cell_indices[0]
        ..=evidence.distance_null_segment_cell_indices[1]];
    let physical_color = Vec3::new(1.0, 0.42, 0.03);
    let distance_null_color = Vec3::new(0.95, 0.05, 0.85);
    let mut vertices =
        Vec::with_capacity((physical_path.len() + distance_null_path.len()).saturating_mul(2) + 80);
    append_route_path_vertices(
        &mut vertices,
        tess,
        hydrology,
        physical_path,
        physical_color,
    );
    append_route_path_vertices(
        &mut vertices,
        tess,
        hydrology,
        distance_null_path,
        distance_null_color,
    );
    for &cell in &evidence.divergence_endpoint_cells {
        append_route_ring_marker(&mut vertices, tess, hydrology, cell, Vec3::ONE);
    }
    append_route_cross_marker(
        &mut vertices,
        tess,
        hydrology,
        evidence.physical_segment.maximum_elevation_cell,
        physical_color,
    );
    append_route_cross_marker(
        &mut vertices,
        tess,
        hydrology,
        evidence.distance_null_segment.maximum_elevation_cell,
        distance_null_color,
    );
    vertices
}

#[allow(clippy::too_many_arguments)]
fn render_relief_with_sites(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_view: &wgpu::TextureView,
    buffers: &super::world::WorldBuffers,
    marker_buffer: &wgpu::Buffer,
    marker_count: u32,
    view: &CaptureView,
    river_mode: RiverMode,
) {
    let uniforms = Uniforms::new(
        view.view_proj,
        view.eye,
        Vec3::new(0.5, 1.0, 0.3).normalize(),
    )
    .with_relief_scale(RELIEF_SCALE)
    .with_slope_shading(true)
    .with_hemisphere_lighting(false)
    .with_map_mode(false)
    .with_rivers(river_mode != RiverMode::Off)
    .with_river_major_only(river_mode == RiverMode::Major)
    .with_river_width_scale(1.0);
    renderer.render_to_view(
        &gpu.device,
        &gpu.queue,
        color_view,
        &uniforms,
        RenderScene {
            fill_pipeline: FillPipelineKind::UnifiedGlobe,
            fill: IndexedDraw {
                vertex_buffer: &buffers.unified_vertex_buffer,
                index_buffer: &buffers.unified_index_buffer,
                index_count: buffers.num_unified_indices,
            },
            river_texture_bind_group: Some(&buffers.river_bind_group),
            edges: None,
            arrows: None,
            pole_markers: None,
            rivers: Some(SurfaceLineDraw {
                vertex_buffer: marker_buffer,
                vertex_count: marker_count,
            }),
            gpu_particles: None,
        },
    );
}

fn run_consequential_geography_packet(opts: &SweepOptions) {
    assert_eq!(
        opts.target_stage, 4,
        "consequential-geography packet requires --stage 4"
    );
    let generation_started = std::time::Instant::now();
    let world = generate_tile_world(opts, &opts.base_erosion);
    let world_generation_ms = generation_started.elapsed().as_secs_f64() * 1_000.0;
    let tess = world.active_tessellation();
    let hydrology = world.active_hydrology().expect("stage 4 hydrology");

    let semantics_started = std::time::Instant::now();
    let water = WaterBodySemantics::build(tess, hydrology);
    let rivers = RiverSelection::build(hydrology, RiverThresholdPolicy::default());
    let living = LivingSurfaceSemantics::build(
        tess,
        world.active_temperature().expect("stage 4 temperature"),
        world.active_precipitation().expect("stage 4 precipitation"),
        hydrology,
    );
    let semantics_build_ms = semantics_started.elapsed().as_secs_f64() * 1_000.0;
    let traversal = TraversalConfig::new(12.0, 3.0).expect("valid probe traversal");
    let component_started = std::time::Instant::now();
    let components = ConsequentialGeographyComponents::build(
        tess, hydrology, &water, &rivers, &living, traversal,
    )
    .expect("build consequential-geography components");
    let component_build_ms = component_started.elapsed().as_secs_f64() * 1_000.0;

    let mut variants = Vec::new();
    let baseline = select_probe_variant(
        "baseline",
        "authored probe baseline; not a product default",
        "none",
        Vec3::new(1.0, 0.90, 0.05),
        &components,
        tess,
        hydrology,
        baseline_site_probe_config(),
        None,
    );
    let baseline_positions = site_positions(tess, &baseline.record.selection);
    variants.push(baseline);
    variants.push(select_probe_variant(
        "diagnostic-coarse-support-160",
        "non-product diagnostic with deliberately under-resolved candidate support",
        "candidate support cap reduced from 512 to 160; every other prior and factor is identical to baseline",
        Vec3::new(0.95, 0.95, 0.95),
        &components,
        tess,
        hydrology,
        diagnostic_coarse_support_site_probe_config(),
        Some(&baseline_positions),
    ));
    variants.push(select_probe_variant(
        "tight-prior",
        "nearby stricter prior panel",
        "tighter spacing, catchment, freshwater, living, grade, area, and coast scales",
        Vec3::new(1.0, 0.15, 0.85),
        &components,
        tess,
        hydrology,
        tight_site_probe_config(),
        Some(&baseline_positions),
    ));
    variants.push(select_probe_variant(
        "loose-prior",
        "nearby more permissive prior panel",
        "looser spacing, catchment, freshwater, living, grade, area, and coast scales",
        Vec3::new(0.05, 0.95, 1.0),
        &components,
        tess,
        hydrology,
        loose_site_probe_config(),
        Some(&baseline_positions),
    ));

    let flat_traversal = TraversalConfig::new(0.0, 0.0).expect("valid flat traversal");
    let grade_component_started = std::time::Instant::now();
    let grade_components = ConsequentialGeographyComponents::build(
        tess,
        hydrology,
        &water,
        &rivers,
        &living,
        flat_traversal,
    )
    .expect("build grade-ablation components");
    let grade_component_build_ms = grade_component_started.elapsed().as_secs_f64() * 1_000.0;
    let mut grade_config = baseline_site_probe_config();
    grade_config.maximum_local_trimmed_mean_grade = f32::MAX;
    variants.push(select_probe_variant(
        "ablate-grade",
        "baseline with traversal and local-grade burden removed",
        "uphill=0, downhill=0, and effectively unbounded local grade gate",
        Vec3::new(1.0, 0.42, 0.05),
        &grade_components,
        tess,
        hydrology,
        grade_config,
        Some(&baseline_positions),
    ));

    let mut freshwater_components = components.clone();
    for cell in 0..tess.num_cells() {
        if hydrology.is_submerged(cell) {
            freshwater_components.freshwater_access_generalized_km[cell] = None;
            freshwater_components.freshwater_source[cell] = false;
            freshwater_components.freshwater_source_kind[cell] = None;
        } else {
            freshwater_components.freshwater_access_generalized_km[cell] = Some(0.0);
            freshwater_components.freshwater_source[cell] = true;
            // The operator requires a source kind. This is an explicitly
            // synthetic mechanics token, not a claim that every cell is river.
            freshwater_components.freshwater_source_kind[cell] =
                Some(FreshwaterSourceKind::SelectedRiver);
        }
    }
    variants.push(select_probe_variant(
        "ablate-freshwater",
        "synthetic uniform freshwater-access null",
        "every land cell is mechanically a zero-burden SelectedRiver source; this token does not represent physical rivers",
        Vec3::new(0.15, 1.0, 0.25),
        &freshwater_components,
        tess,
        hydrology,
        baseline_site_probe_config(),
        Some(&baseline_positions),
    ));

    let mut coast_config = baseline_site_probe_config();
    coast_config.coast_bonus = 0.0;
    variants.push(select_probe_variant(
        "ablate-coast",
        "baseline with coast preference removed",
        "coast bonus set to zero; coast access retained for provenance",
        Vec3::new(1.0, 0.12, 0.12),
        &components,
        tess,
        hydrology,
        coast_config,
        Some(&baseline_positions),
    ));

    let mut living_components = components.clone();
    for cell in 0..tess.num_cells() {
        living_components.relative_living_opportunity[cell] = if hydrology.is_submerged(cell) {
            0.0
        } else {
            1.0
        };
    }
    variants.push(select_probe_variant(
        "ablate-living",
        "synthetic uniform land living-opportunity null",
        "all land opportunity set to 1.0; water set to 0.0",
        Vec3::new(0.70, 0.25, 1.0),
        &living_components,
        tess,
        hydrology,
        baseline_site_probe_config(),
        Some(&baseline_positions),
    ));

    let route_site_selection = variants[0].record.selection.clone();
    let route_config = route_probe_config();
    let physical_route_started = Instant::now();
    let physical_route_network = build_aggregate_route_network(
        tess,
        hydrology,
        &route_site_selection,
        traversal,
        route_config,
    )
    .expect("build physical-terrain aggregate route probe");
    let physical_route_build_ms = physical_route_started.elapsed().as_secs_f64() * 1_000.0;
    let zero_grade_route_started = Instant::now();
    let zero_grade_route_network = build_aggregate_route_network(
        tess,
        hydrology,
        &route_site_selection,
        flat_traversal,
        route_config,
    )
    .expect("build zero-grade aggregate route counterfactual");
    let zero_grade_route_build_ms = zero_grade_route_started.elapsed().as_secs_f64() * 1_000.0;
    let route_comparison =
        compare_route_networks(tess, &physical_route_network, &zero_grade_route_network);
    assert!(
        route_comparison.identical_site_anchor_input,
        "route counterfactual must preserve exact site anchors"
    );
    let mut lower_corridor_probe = lower_corridor_probe_record(
        tess,
        hydrology,
        &physical_route_network,
        &zero_grade_route_network,
    )
    .expect("assess route-local lower-terrain corridor");
    let mut route_variants = vec![
        RouteProbeVariant {
            id: "route-physical-terrain",
            role: "bounded terrestrial network using the direction-neutral half-round-trip average of disclosed directional grade costs",
            line_color_linear_rgb: [1.0, 0.42, 0.03],
            build_ms: physical_route_build_ms,
            image_filenames: Vec::new(),
            network: physical_route_network,
            line_color: Vec3::new(1.0, 0.42, 0.03),
        },
        RouteProbeVariant {
            id: "route-zero-grade",
            role: "matched-site counterfactual with uphill and downhill burden set to zero",
            line_color_linear_rgb: [0.95, 0.05, 0.85],
            build_ms: zero_grade_route_build_ms,
            image_filenames: Vec::new(),
            network: zero_grade_route_network,
            line_color: Vec3::new(0.95, 0.05, 0.85),
        },
    ];

    let views = consequential_probe_views(
        opts,
        tess,
        &route_site_selection,
        route_comparison.most_divergent_selected_target_cell,
    );
    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|error| panic!("create {}: {error}", opts.out_dir.display()));
    let gpu = pollster::block_on(GpuContext::new_headless(opts.width, opts.height));
    let mut renderer = Renderer::new(&gpu, &Uniforms::new(Mat4::IDENTITY, Vec3::ZERO, Vec3::Y));
    let color_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("consequential_geography_color"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&Default::default());
    let world_buffers = generate_world_buffers(&gpu.device, &gpu.queue, &world);
    let montage_width = opts.width * views.len() as u32;
    let montage_height = opts.height * (variants.len() + route_variants.len()) as u32;
    let mut montage = vec![0; (montage_width * montage_height * 4) as usize];
    for (row, variant) in variants.iter_mut().enumerate() {
        let markers = site_marker_vertices(
            tess,
            hydrology,
            &variant.record.selection,
            variant.marker_color,
        );
        let marker_buffer = create_vertex_buffer(
            &gpu.device,
            &markers,
            "consequential_geography_site_markers",
        );
        for (column, view) in views.iter().enumerate() {
            render_relief_with_sites(
                &gpu,
                &mut renderer,
                &color_view,
                &world_buffers,
                &marker_buffer,
                markers.len() as u32,
                view,
                opts.river_mode,
            );
            let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
            let filename = format!("{:02}_{}_{}.png", row + 1, variant.record.id, view.label);
            write_png(
                &opts.out_dir.join(&filename),
                &rgba,
                opts.width,
                opts.height,
            );
            blit_tile(
                &mut montage,
                montage_width,
                &rgba,
                opts.width,
                opts.height,
                column as u32,
                row as u32,
            );
            variant.record.image_filenames.push(filename);
        }
    }
    for (route_index, variant) in route_variants.iter_mut().enumerate() {
        let row = variants.len() + route_index;
        let overlay = route_overlay_vertices(
            tess,
            hydrology,
            &route_site_selection,
            &variant.network,
            variant.line_color,
            Vec3::new(1.0, 0.90, 0.05),
        );
        let overlay_buffer = create_vertex_buffer(
            &gpu.device,
            &overlay,
            "consequential_geography_route_overlay",
        );
        for (column, view) in views.iter().enumerate() {
            render_relief_with_sites(
                &gpu,
                &mut renderer,
                &color_view,
                &world_buffers,
                &overlay_buffer,
                overlay.len() as u32,
                view,
                opts.river_mode,
            );
            let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
            let filename = format!("{:02}_{}_{}.png", row + 1, variant.id, view.label);
            write_png(
                &opts.out_dir.join(&filename),
                &rgba,
                opts.width,
                opts.height,
            );
            blit_tile(
                &mut montage,
                montage_width,
                &rgba,
                opts.width,
                opts.height,
                column as u32,
                row as u32,
            );
            variant.image_filenames.push(filename);
        }
    }
    if let Some(evidence) = lower_corridor_probe.strongest.clone() {
        let physical_cell = evidence.maximum_symmetric_cell_center_separation_cells[0];
        let distance_null_cell = evidence.maximum_symmetric_cell_center_separation_cells[1];
        let physical_position = tess.cell_center(physical_cell);
        let distance_null_position = tess.cell_center(distance_null_cell);
        let midpoint = physical_position + distance_null_position;
        let target = if midpoint.length_squared() > 1.0e-8 {
            midpoint.normalize()
        } else {
            physical_position
        };
        let mut view = derived_capture_view(
            "route-local-lower-terrain-corridor",
            target,
            opts.width as f32 / opts.height as f32,
            opts.zoom_alt,
        );
        view.sidecar.kind = "derived-lower-terrain-corridor";
        let overlay = lower_corridor_overlay_vertices(
            tess,
            hydrology,
            &route_variants[0].network,
            &route_variants[1].network,
            &evidence,
        );
        let overlay_buffer = create_vertex_buffer(
            &gpu.device,
            &overlay,
            "consequential_geography_lower_corridor_overlay",
        );
        render_relief_with_sites(
            &gpu,
            &mut renderer,
            &color_view,
            &world_buffers,
            &overlay_buffer,
            overlay.len() as u32,
            &view,
            opts.river_mode,
        );
        let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
        let filename = "route-local-lower-terrain-corridor.png".to_string();
        write_png(
            &opts.out_dir.join(&filename),
            &rgba,
            opts.width,
            opts.height,
        );
        lower_corridor_probe.render = Some(RouteLowerCorridorRenderRecord {
            filename,
            camera: view.sidecar,
            physical_branch_color_linear_rgb: [1.0, 0.42, 0.03],
            distance_null_branch_color_linear_rgb: [0.95, 0.05, 0.85],
            divergence_endpoint_color_linear_rgb: [1.0, 1.0, 1.0],
            physical_crest_color_linear_rgb: [1.0, 0.42, 0.03],
            distance_null_crest_color_linear_rgb: [0.95, 0.05, 0.85],
        });
    }
    write_png(
        &opts.out_dir.join("montage.png"),
        &montage,
        montage_width,
        montage_height,
    );

    let sidecar = serde_json::json!({
        "schema_version": 3,
        "purpose": "discriminating evaluation packet for bounded aggregate Consequential Geography site selection, a same-site terrestrial route counterfactual, and one conservative lower-terrain-corridor explanation",
        "status": "implemented on-demand site and route probe; neither model is promoted, a product default, persistent World state, or Stage 5",
        "world_manifest": world.manifest(),
        "timing_ms": {
            "single_stage_4_world_generation": world_generation_ms,
            "water_river_living_semantics": semantics_build_ms,
            "baseline_access_components": component_build_ms,
            "grade_ablation_access_components": grade_component_build_ms
        },
        "shared_component_config": {
            "river_policy": RiverThresholdPolicy::default(),
            "traversal": traversal
        },
        "truth_contract": [
            "all variants use one unchanged Stage-4 physical world and matched cameras",
            "both route rows use the exact same baseline site anchors and candidate endpoint pairs; only traversal grade burden changes",
            "route selection uses the direction-neutral half-round-trip average of directional uphill and downhill edge costs; the directional asymmetry remains evidence but does not choose a direction-specific path",
            "a lower-terrain-corridor explanation compares elementary physical and distance-null branches with the exact same split and rejoin cells and is omitted unless the physical branch is longer, cheaper under physical traversal, and lower at its maximum elevation",
            "route geometry is a bounded land-only aggregate network, not roads, maritime travel, travel time, settlement history, or an optimized product graph",
            "the 160-candidate coarse-support arm is a deliberately under-resolved diagnostic comparator, not a proposed product configuration",
            "sites are deterministic aggregate opportunity anchors, not settlements, population, culture, resources, ownership, or persistent identities",
            "freshwater means selected aggregate rivers or proper-lake shores; coast means semantic ocean coast",
            "living opportunity is accepted Living Surface vegetation cover, not productivity, yield, or carrying capacity",
            "relief and river styling are presentation only; selection uses physical Stage-4 elevations and disclosed generalized costs",
            "the freshwater ablation's SelectedRiver source kind is only an operator-compatible synthetic token and does not claim ubiquitous rivers"
        ],
        "known_limitations": [
            "the baseline, tight, and loose configurations are an authored diagnostic prior panel, not fitted or calibrated",
            "the 512-candidate baseline is still a bounded maximin support, not exhaustive evaluation of every eligible cell",
            "neutral-score ties remain cell-ID-dependent",
            "nearest-site comparison is directional from each variant anchor to the baseline set and does not solve optimal bipartite matching",
            "marker rings and crosses have a fixed angular size and are cartographic annotations, not site extent",
            "the regional camera targets the largest selected-route path divergence when one exists, not a human-curated geography",
            "route strokes are one-pixel evidence overlays and do not communicate hierarchy, reuse, crossings, passes, chokepoints, or construction",
            "the lower-corridor comparator is the zero-grade distance-minimizing branch, not an enumerated second-best physical route, and it does not establish a geomorphic gap, pass, ridge or barrier",
            "drainage-repaired support length counts a full route edge when either endpoint was repaired; it is a conservative support proxy rather than geometric overlap length",
            "candidate route pairs come from a physical spanning tree plus bounded nearest neighbors; they are not exhaustive",
            "no route-derived labels, factor rasters, population, economy, or culture are generated"
        ],
        "comparison_definition": {
            "exact_retained_anchor": "same Voronoi anchor cell appears in baseline and variant",
            "nearest_distance": "great-circle physical kilometres from each variant anchor to its nearest baseline anchor",
            "median": "ordinary sample median of directional nearest distances",
            "route_endpoint_jaccard": "set Jaccard over selected site-ID endpoint pairs",
            "route_cell_edge_jaccard": "set Jaccard over undirected Voronoi cell edges traversed by selected routes",
            "candidate_path_hausdorff": "symmetric cell-centre-sampled Hausdorff distance in great-circle kilometres between physical and zero-grade route samples for the same candidate endpoints; it is not continuous polyline distance"
        },
        "cameras": views.iter().map(|view| view.sidecar.clone()).collect::<Vec<_>>(),
        "variants": variants.into_iter().map(|variant| variant.record).collect::<Vec<_>>(),
        "route_probe": {
            "config": route_config,
            "fixed_site_anchor_cells": route_site_selection.sites.iter().map(|site| site.anchor_cell).collect::<Vec<_>>(),
            "comparison": route_comparison,
            "lower_corridor_explanation": lower_corridor_probe,
            "rows": route_variants
        },
        "montage": {
            "filename": "montage.png",
            "rows": "site variants in sidecar order, followed by route_probe rows in sidecar order",
            "columns": "cameras in sidecar order"
        }
    });
    let sidecar_path = opts.out_dir.join("consequential-geography.json");
    let file = std::fs::File::create(&sidecar_path)
        .unwrap_or_else(|error| panic!("create {}: {error}", sidecar_path.display()));
    serde_json::to_writer_pretty(BufWriter::new(file), &sidecar)
        .expect("write consequential-geography.json");
    println!(
        "Done: consequential-geography probe packet -> {}",
        opts.out_dir.display()
    );
}

#[derive(Clone, Debug, Serialize)]
struct ReadabilityRelationship {
    selection_rule: &'static str,
    fallback: &'static str,
    mouth_cell: usize,
    downstream_water_cell: usize,
    head_cell: usize,
    trunk_cells_head_to_mouth: Vec<usize>,
    discharge_equivalent_km2: f32,
    strahler_order_at_mouth: u8,
    selected_mouth_is_major: bool,
    receiving_water_body_index: usize,
    receiving_water_body_id: WaterBodyId,
    receiving_water_kind: SemanticWaterKind,
    receiving_water_area_km2: f32,
    receiving_water_anchor_cell: usize,
}

#[derive(Clone, Copy, Debug)]
struct ReadabilityMouthCandidate {
    mouth_cell: usize,
    downstream_water_cell: usize,
    water_body_index: usize,
    discharge: f32,
    major: bool,
}

/// Choose an ocean/lake mouth without letting iterator order decide equal-flow
/// ties. A world with no major candidate falls back to all represented mouths.
fn select_readability_mouth(
    candidates: &[ReadabilityMouthCandidate],
) -> Option<(ReadabilityMouthCandidate, &'static str)> {
    let best = |major_only: bool| {
        candidates
            .iter()
            .filter(|candidate| !major_only || candidate.major)
            .max_by(|a, b| {
                a.discharge
                    .total_cmp(&b.discharge)
                    // Lower cell identity wins an exact discharge tie.
                    .then_with(|| b.mouth_cell.cmp(&a.mouth_cell))
            })
            .copied()
    };
    best(true)
        .map(|candidate| (candidate, "none"))
        .or_else(|| {
            best(false).map(|candidate| {
                (
                    candidate,
                    "no major river mouth reached a semantic ocean or lake; selected the highest-discharge represented river mouth reaching an ocean or lake",
                )
            })
        })
}

fn readability_main_trunk(
    mouth: usize,
    hydrology: &hex3::world::Hydrology,
    network: &RiverNetwork,
) -> Vec<usize> {
    let mut trunk = vec![mouth];
    let mut current = mouth;
    while let Some(&upstream) = network.upstream[current].iter().max_by(|&&a, &&b| {
        hydrology.flow_accumulation[a]
            .total_cmp(&hydrology.flow_accumulation[b])
            // Lower cell identity wins an exact discharge tie.
            .then_with(|| b.cmp(&a))
    }) {
        trunk.push(upstream);
        current = upstream;
        assert!(
            trunk.len() <= network.all_cells.len(),
            "river trunk unexpectedly contains a drainage cycle"
        );
    }
    trunk.reverse();
    trunk
}

fn readability_relationship(
    hydrology: &hex3::world::Hydrology,
    water: &WaterBodySemantics,
    network: &RiverNetwork,
) -> ReadabilityRelationship {
    let candidates: Vec<_> = network
        .mouths
        .iter()
        .filter_map(|&mouth_cell| {
            let downstream_water_cell = hydrology.downstream(mouth_cell)?;
            let water_body_index = water.cell_body[downstream_water_cell]?;
            if !matches!(
                water.bodies[water_body_index].kind,
                SemanticWaterKind::Ocean | SemanticWaterKind::Lake
            ) {
                return None;
            }
            Some(ReadabilityMouthCandidate {
                mouth_cell,
                downstream_water_cell,
                water_body_index,
                discharge: hydrology.flow_accumulation[mouth_cell],
                major: network.major_cells[mouth_cell],
            })
        })
        .collect();
    let (selected, fallback) = select_readability_mouth(&candidates).expect(
        "world-readability-v0 requires at least one represented river mouth reaching a semantic ocean or lake",
    );
    let trunk_cells_head_to_mouth = readability_main_trunk(selected.mouth_cell, hydrology, network);
    let body = &water.bodies[selected.water_body_index];
    ReadabilityRelationship {
        selection_rule: "highest flow_accumulation among represented major river mouths reaching a semantic ocean or lake; exact ties choose lower mouth cell identity",
        fallback,
        mouth_cell: selected.mouth_cell,
        downstream_water_cell: selected.downstream_water_cell,
        head_cell: trunk_cells_head_to_mouth[0],
        trunk_cells_head_to_mouth,
        discharge_equivalent_km2: selected.discharge * PLANET_RADIUS_KM.powi(2),
        strahler_order_at_mouth: network.strahler_order[selected.mouth_cell],
        selected_mouth_is_major: selected.major,
        receiving_water_body_index: selected.water_body_index,
        receiving_water_body_id: body.id,
        receiving_water_kind: body.kind,
        receiving_water_area_km2: body.area_km2,
        receiving_water_anchor_cell: body.id.anchor_cell,
    }
}

#[derive(Clone, Debug, Serialize)]
struct ReadabilityStateCounts {
    active_cells: usize,
    drainage_edges: usize,
    semantic_water_bodies: usize,
    semantic_water_owned_cells: usize,
    all_river_cells: usize,
    major_river_cells: usize,
    river_mouths: usize,
    river_reaches: usize,
    trunk_cells: usize,
}

fn readability_state_evidence(
    world: &World,
    policy: RiverThresholdPolicy,
) -> (ReadabilityStateCounts, ReadabilityRelationship) {
    let tessellation = world.active_tessellation();
    let hydrology = world.active_hydrology().expect("stage 4 hydrology");
    let water = WaterBodySemantics::build(tessellation, hydrology);
    let network = RiverNetwork::build(tessellation, hydrology, &water, policy);
    let relationship = readability_relationship(hydrology, &water, &network);
    let counts = ReadabilityStateCounts {
        active_cells: tessellation.num_cells(),
        drainage_edges: hydrology.drainage_dir.iter().flatten().count(),
        semantic_water_bodies: water.bodies.len(),
        semantic_water_owned_cells: water.cell_body.iter().flatten().count(),
        all_river_cells: network
            .all_cells
            .iter()
            .filter(|&&included| included)
            .count(),
        major_river_cells: network
            .major_cells
            .iter()
            .filter(|&&included| included)
            .count(),
        river_mouths: network.mouths.len(),
        river_reaches: network.reaches.len(),
        trunk_cells: relationship.trunk_cells_head_to_mouth.len(),
    };
    (counts, relationship)
}

fn readability_river_policy(opts: &SweepOptions) -> RiverThresholdPolicy {
    match opts.river_threshold_policy.as_str() {
        "legacy-count-equivalent" => RiverThresholdPolicy::legacy(),
        "catchment-km2" => RiverThresholdPolicy::catchment(
            opts.river_min_catchment_km2
                .unwrap_or(hex3::world::DEFAULT_RIVER_MIN_CATCHMENT_KM2),
        ),
        other => panic!("world-readability-v0 does not recognize river policy '{other}'"),
    }
}

#[derive(Clone, Debug, Serialize)]
struct ReadabilityViewRecord {
    id: &'static str,
    role: &'static str,
    projection: &'static str,
    map_mode: bool,
    camera: serde_json::Value,
    physical_relief_effect: &'static str,
}

struct ReadabilityView {
    record: ReadabilityViewRecord,
    view_proj: Mat4,
    eye: Vec3,
}

fn readability_views(
    opts: &SweepOptions,
    world: &World,
    relationship: &ReadabilityRelationship,
    aspect: f32,
) -> Vec<ReadabilityView> {
    let mut globe = OrbitCamera::new();
    globe.yaw = opts.yaw_deg.to_radians();
    globe.pitch = opts.pitch_deg.to_radians();
    globe.distance = opts.distance;
    globe.aspect = aspect;
    let globe_eye = globe.eye_position();

    let map_projection = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    let tessellation = world.active_tessellation();
    let lower_trunk_index = relationship
        .trunk_cells_head_to_mouth
        .len()
        .saturating_sub(1)
        * 3
        / 4;
    let lower_trunk =
        tessellation.cell_center(relationship.trunk_cells_head_to_mouth[lower_trunk_index]);
    let mouth = tessellation.cell_center(relationship.mouth_cell);
    let receiving = tessellation.cell_center(relationship.downstream_water_cell);
    let regional_target = (lower_trunk + mouth * 2.0 + receiving).normalize_or_zero();
    assert!(
        regional_target.length_squared() > 0.0,
        "derived relationship target is degenerate"
    );
    let mut regional = derived_capture_view(
        "river-mouth-relationship",
        regional_target,
        aspect,
        opts.zoom_alt,
    );
    regional.sidecar.kind = "derived-highest-discharge-river-mouth";

    vec![
        ReadabilityView {
            record: ReadabilityViewRecord {
                id: "globe",
                role: "fixed whole-planet overview shared by both rows",
                projection: "perspective globe",
                map_mode: false,
                camera: serde_json::json!({
                    "eye_xyz": vec3_array(globe_eye),
                    "aim_xyz": [0.0, 0.0, 0.0],
                    "up_xyz": [0.0, 1.0, 0.0],
                    "vertical_fov_deg": 45.0,
                    "aspect": aspect,
                    "near": 0.01,
                    "far": 10.0,
                    "orbit_yaw_deg": opts.yaw_deg,
                    "orbit_pitch_deg": opts.pitch_deg,
                    "orbit_distance": opts.distance,
                }),
                physical_relief_effect: "Authentic radial displacement and slope shading",
            },
            view_proj: globe.view_projection(),
            eye: globe_eye,
        },
        ReadabilityView {
            record: ReadabilityViewRecord {
                id: "map",
                role: "fixed full-world map shared by both rows",
                projection: "equirectangular, longitude=atan2(z,x), latitude=asin(y), exact 2:1 viewport",
                map_mode: true,
                camera: serde_json::json!({
                    "orthographic_bounds": [-1.0, 1.0, -1.0, 1.0],
                    "aspect": aspect,
                    "eye_xyz": [0.0, 0.0, 1.0],
                }),
                physical_relief_effect: "flat projection: the map shader intentionally ignores radial relief displacement; Authentic remains the declared row recipe, not effective map geometry",
            },
            view_proj: map_projection,
            eye: Vec3::Z,
        },
        ReadabilityView {
            record: ReadabilityViewRecord {
                id: "river-mouth-relationship",
                role: "derived regional view of the selected trunk entering its semantic receiving water body",
                projection: "perspective globe close-up",
                map_mode: false,
                camera: serde_json::to_value(&regional.sidecar).expect("serialize regional camera"),
                physical_relief_effect: "Authentic radial displacement and slope shading",
            },
            view_proj: regional.view_proj,
            eye: regional.eye,
        },
    ]
}

#[allow(clippy::too_many_arguments)]
fn render_readability_view(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_view: &wgpu::TextureView,
    buffers: &super::world::WorldBuffers,
    view: &ReadabilityView,
    river_mode: RiverMode,
    river_width_scale: f32,
) {
    let light = if view.record.map_mode {
        Vec3::Z
    } else {
        Vec3::new(0.5, 1.0, 0.3).normalize()
    };
    let uniforms = Uniforms::new(view.view_proj, view.eye, light)
        .with_relief_scale(ReliefPreset::Authentic.scale())
        .with_slope_shading(!view.record.map_mode)
        .with_hemisphere_lighting(false)
        .with_map_mode(view.record.map_mode)
        .with_rivers(true)
        .with_river_major_only(river_mode == RiverMode::Major)
        .with_river_width_scale(river_width_scale);
    renderer.render_to_view(
        &gpu.device,
        &gpu.queue,
        color_view,
        &uniforms,
        RenderScene {
            fill_pipeline: if view.record.map_mode {
                FillPipelineKind::UnifiedMap
            } else {
                FillPipelineKind::UnifiedGlobe
            },
            fill: IndexedDraw {
                vertex_buffer: &buffers.unified_vertex_buffer,
                index_buffer: &buffers.unified_index_buffer,
                index_count: buffers.num_unified_indices,
            },
            river_texture_bind_group: Some(&buffers.river_bind_group),
            edges: None,
            arrows: None,
            pole_markers: None,
            rivers: None,
            gpu_particles: None,
        },
    );
}

/// Minimal product-readability discriminator. It owns no new model: two
/// presentation recipes consume one immutable Stage-4 world and one river SDF.
fn run_world_readability_packet(opts: &SweepOptions) {
    assert_eq!(
        opts.target_stage, 4,
        "world-readability-v0 requires --stage 4"
    );
    assert_eq!(
        opts.display_subdivision_levels, 0,
        "world-readability-v0 compares current product rendering without display subdivision"
    );
    let world = generate_tile_world(opts, &opts.base_erosion);
    let policy = readability_river_policy(opts);
    let (state_counts, relationship) = readability_state_evidence(&world, policy);

    // An exact 2:1 target makes the full map equirectangular. Packet width is
    // rounded up to even; the generic sweep height is intentionally ignored.
    let width = opts.width.max(2).next_multiple_of(2);
    let height = width / 2;
    let aspect = width as f32 / height as f32;
    let views = readability_views(opts, &world, &relationship, aspect);
    let river_width_scale = opts.base_erosion.river_width_scale.unwrap_or(1.0);

    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|error| panic!("create {}: {error}", opts.out_dir.display()));
    let gpu = pollster::block_on(GpuContext::new_headless(width, height));
    let mut renderer = Renderer::new(&gpu, &Uniforms::new(Mat4::IDENTITY, Vec3::ZERO, Vec3::Y));
    let color_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("world_readability_v0_color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&Default::default());
    let mut buffers = generate_world_buffers(&gpu.device, &gpu.queue, &world);
    assert_eq!(buffers.surface_palette, SurfacePalette::Terrain);
    let unified_index_count = buffers.num_unified_indices;
    let montage_width = width * views.len() as u32;
    let montage_height = height * 2;
    let mut montage = vec![0; (montage_width * montage_height * 4) as usize];
    let row_recipes = [
        (
            "control-terrain",
            "current control: Terrain palette, Authentic relief, Major rivers in all views",
            SurfacePalette::Terrain,
        ),
        (
            "candidate-living-surface",
            "candidate composed presentation recipe: existing Living Surface palette, Authentic relief, Major rivers globally and All rivers in the regional relationship view",
            SurfacePalette::LivingSurface,
        ),
    ];
    let mut row_records = Vec::new();
    for (row, (id, role, palette)) in row_recipes.into_iter().enumerate() {
        regenerate_surface_palette(&gpu.queue, &world, &mut buffers, palette)
            .unwrap_or_else(|error| panic!("world-readability-v0 palette: {error}"));
        assert_eq!(
            buffers.num_unified_indices, unified_index_count,
            "presentation palette changed unified topology"
        );
        let mut image_records = Vec::new();
        for (column, view) in views.iter().enumerate() {
            let river_mode = if row == 1 && view.record.id == "river-mouth-relationship" {
                RiverMode::All
            } else {
                RiverMode::Major
            };
            render_readability_view(
                &gpu,
                &mut renderer,
                &color_view,
                &buffers,
                view,
                river_mode,
                river_width_scale,
            );
            let rgba = read_back_rgba(&gpu, &color_tex, width, height);
            let filename = format!("{:02}_{id}_{}.png", row + 1, view.record.id);
            write_png(&opts.out_dir.join(&filename), &rgba, width, height);
            blit_tile(
                &mut montage,
                montage_width,
                &rgba,
                width,
                height,
                column as u32,
                row as u32,
            );
            image_records.push(serde_json::json!({
                "view_id": view.record.id,
                "filename": filename,
                "river_mode": river_mode_label(river_mode),
                "relief_preset": ReliefPreset::Authentic.name(),
                "relief_scale": ReliefPreset::Authentic.scale(),
                "river_width_scale": river_width_scale,
                "surface_palette": palette.name(),
            }));
        }
        row_records.push(serde_json::json!({
            "id": id,
            "role": role,
            "surface_palette": palette.name(),
            "relief_preset": ReliefPreset::Authentic.name(),
            "relief_scale": ReliefPreset::Authentic.scale(),
            "images": image_records,
        }));
    }
    write_png(
        &opts.out_dir.join("montage.png"),
        &montage,
        montage_width,
        montage_height,
    );

    let sidecar = serde_json::json!({
        "schema_version": 1,
        "purpose": "minimal World Readability V0 comparison over one unchanged Stage-4 world",
        "status": "headless presentation discriminator; no new physical or semantic model and no promoted default",
        "world_manifest": world.manifest(),
        "truth_contract": [
            "both rows use the same immutable Stage-4 World and the same WorldBuffers topology and river texture",
            "Terrain to Living Surface changes only baked unified-vertex colors; regenerate_surface_palette verifies vertex and index counts",
            "globe and map use the same Major mask and the same semantic relationship identities",
            "candidate regional All rivers is an authored scale-dependent presentation policy, so this packet is a composed recipe comparison rather than a palette-only ablation",
            "water classification, river selection, relief and world state are not regenerated between rows"
        ],
        "viewport": {
            "requested_sweep_width": opts.width,
            "requested_sweep_height_ignored": opts.height,
            "effective_width": width,
            "effective_height": height,
            "reason": "full equirectangular map requires an exact 2:1 target; width is rounded up to even"
        },
        "river_policy": {
            "requested_name": opts.river_threshold_policy,
            "requested_minimum_catchment_km2": opts.river_min_catchment_km2,
            "effective_policy": policy,
            "effective_all_minimum_catchment_km2": policy.effective_all_minimum_km2(world.num_cells()),
            "selection_mask_counts": {
                "all": state_counts.all_river_cells,
                "major": state_counts.major_river_cells,
            }
        },
        "selected_relationship": relationship,
        "shared_state_evidence": {
            "exact_counts": state_counts,
            "one_immutable_world": true,
            "one_shared_world_buffers_instance": true,
            "one_shared_river_texture": true,
            "unified_index_count_preserved_across_palette_regeneration": unified_index_count,
        },
        "views": views.iter().map(|view| &view.record).collect::<Vec<_>>(),
        "rows": row_records,
        "montage": {
            "filename": "montage.png",
            "rows": "control Terrain recipe, then candidate Living Surface recipe",
            "columns": "globe, equirectangular map, derived river-mouth relationship"
        },
        "known_limitations": [
            "Living Surface is an existing equilibrium physiognomy presentation, not persistent ecology or a biome model",
            "the equirectangular map is intentionally flat: Authentic relief is not geometric in map mode",
            "the regional camera is mechanically derived from the lower trunk, mouth and first receiving-water cell; it is not human-curated",
            "if no represented major river reaches a semantic ocean or lake, selection falls back to the highest-discharge represented river reaching an ocean or lake and records that fact",
            "the candidate changes both palette and regional river density, so it cannot isolate their individual visual contributions",
            "this packet does not tune colors, widths, relief, water classification or renderer behavior"
        ]
    });
    let file = std::fs::File::create(opts.out_dir.join("world-readability-v0.json"))
        .expect("create world-readability-v0.json");
    serde_json::to_writer_pretty(BufWriter::new(file), &sidecar)
        .expect("write world-readability-v0.json");
    println!(
        "Done: World Readability V0 packet -> {}",
        opts.out_dir.display()
    );
}

#[cfg(feature = "research-landscape")]
const RDS0_TERRAIN_SEED: u64 = 8_675_309;
#[cfg(feature = "research-landscape")]
const RDS0_TERRAIN_COARSE_CELLS: usize = 100_000;
#[cfg(feature = "research-landscape")]
const RDS0_TERRAIN_FINE_CAP: usize = 250_000;
#[cfg(feature = "research-landscape")]
const RDS0_TERRAIN_EPISODE: usize = 9;
#[cfg(feature = "research-landscape")]
const RDS0_TERRAIN_ARM_COUNT: usize = 3;

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct Rds0SurfaceSummary {
    supplied_elevation_fnv1a64: String,
    hydrology_elevation_fnv1a64: String,
    drainage_fnv1a64: String,
    minimum_elevation: f32,
    median_elevation: f32,
    p95_elevation: f32,
    maximum_elevation: f32,
    positive_elevation_cell_fraction: f64,
    ocean_cell_fraction: f64,
    lake_count: usize,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct Rds0SurfaceDelta {
    mean_absolute_elevation_delta: f64,
    root_mean_square_elevation_delta: f64,
    maximum_absolute_elevation_delta: f32,
    positive_delta_cell_fraction: f64,
    support_mean_absolute_elevation_delta: f64,
    support_root_mean_square_elevation_delta: f64,
    support_maximum_absolute_elevation_delta: f32,
    support_drainage_receiver_changed_fraction: f64,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct Rds0LocalSurfaceSummary {
    cell_count: usize,
    cell_fraction: f64,
    spherical_area_fraction: f64,
    area_weighted_mean_elevation: f64,
    area_weighted_elevation_std_dev: f64,
    maximum_elevation: f32,
    median_max_downhill_slope: f32,
    p95_max_downhill_slope: f32,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct Rds0RenderRecord {
    arm: &'static str,
    presentation: &'static str,
    relief_scale: f32,
    image_filenames: Vec<String>,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct B0ScaffoldFinalAuditV0 {
    scaffold_receiver_fnv1a64: String,
    promoted_channel_fnv1a64: String,
    strahler_order_fnv1a64: String,
    scaffold_terminal_count: usize,
    final_receiver_agreement_fraction: f64,
    support_receiver_agreement_fraction: f64,
    promoted_channel_receiver_agreement_fraction: f64,
    mismatch_cell_count: usize,
    montage_filename: &'static str,
    montage_column_order: [&'static str; 2],
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct Rds0UnrepresentedRasterReport {
    field: &'static str,
    active_unrepresented_owner_count: usize,
    requested_unrepresented_opportunity_km2_per_myr: f64,
    owner_id_sample: Vec<usize>,
    owner_id_sample_truncated: bool,
}

#[cfg(feature = "research-landscape")]
#[derive(Debug, Serialize)]
struct Rds0TransferPreflight {
    schema: &'static str,
    coarse_owner_count: usize,
    represented_coarse_owner_count: usize,
    unrepresented_coarse_owner_count: usize,
    active_owner_sample_limit: usize,
    rasters: Vec<Rds0UnrepresentedRasterReport>,
    center_owner_transfer_feasible: bool,
}

#[cfg(feature = "research-landscape")]
fn rds0_transfer_preflight(
    coarse: &Tessellation,
    fine_coarse_owner: &[usize],
    control: &RegionalDeformationRasterV0,
    frames: &[RegionalDeformationRasterV0],
) -> Rds0TransferPreflight {
    const SAMPLE_LIMIT: usize = 32;
    let mut represented = vec![false; coarse.num_cells()];
    for &owner in fine_coarse_owner {
        assert!(
            owner < represented.len(),
            "fine owner outside coarse domain"
        );
        represented[owner] = true;
    }
    let areas = coarse.cell_areas_ref();
    let physical_area_scale = f64::from(PLANET_RADIUS_KM).powi(2);
    let report = |field: &'static str, raster: &RegionalDeformationRasterV0| {
        assert_eq!(raster.rate_density_per_myr.len(), coarse.num_cells());
        let active_ids: Vec<usize> = (0..coarse.num_cells())
            .filter(|&cell| !represented[cell] && raster.rate_density_per_myr[cell] > 0.0)
            .collect();
        let requested = active_ids
            .iter()
            .map(|&cell| {
                raster.rate_density_per_myr[cell] * f64::from(areas[cell]) * physical_area_scale
            })
            .sum();
        Rds0UnrepresentedRasterReport {
            field,
            active_unrepresented_owner_count: active_ids.len(),
            requested_unrepresented_opportunity_km2_per_myr: requested,
            owner_id_sample: active_ids.iter().take(SAMPLE_LIMIT).copied().collect(),
            owner_id_sample_truncated: active_ids.len() > SAMPLE_LIMIT,
        }
    };
    let mut rasters = Vec::with_capacity(1 + frames.len());
    rasters.push(report("static-control", control));
    for (index, frame) in frames.iter().enumerate() {
        rasters.push(report(
            ["frame-0", "frame-1", "frame-2", "frame-3"][index],
            frame,
        ));
    }
    let represented_count = represented.iter().filter(|&&owner| owner).count();
    let center_owner_transfer_feasible = rasters
        .iter()
        .all(|raster| raster.active_unrepresented_owner_count == 0);
    Rds0TransferPreflight {
        schema: "hex3.rds0-transfer-preflight.v0",
        coarse_owner_count: coarse.num_cells(),
        represented_coarse_owner_count: represented_count,
        unrepresented_coarse_owner_count: coarse.num_cells() - represented_count,
        active_owner_sample_limit: SAMPLE_LIMIT,
        rasters,
        center_owner_transfer_feasible,
    }
}

#[cfg(feature = "research-landscape")]
fn rds0_fnv1a64(chunks: impl IntoIterator<Item = u64>) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in chunks {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

#[cfg(feature = "research-landscape")]
fn rds0_surface_summary(surface: &FineSurface) -> Rds0SurfaceSummary {
    let values = &surface.elevation.values;
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    assert_eq!(
        finite.len(),
        values.len(),
        "RDS0 terrain emitted non-finite elevation"
    );
    finite.sort_unstable_by(f32::total_cmp);
    let at = |fraction: f64| finite[((finite.len() - 1) as f64 * fraction).round() as usize];
    let drainage_words = surface
        .hydrology
        .drainage_dir
        .iter()
        .map(|next| next.map_or(u64::MAX, |cell| cell as u64));
    Rds0SurfaceSummary {
        supplied_elevation_fnv1a64: rds0_fnv1a64(
            values.iter().map(|value| u64::from(value.to_bits())),
        ),
        hydrology_elevation_fnv1a64: rds0_fnv1a64(
            surface
                .hydrology
                .elevation
                .iter()
                .map(|value| u64::from(value.to_bits())),
        ),
        drainage_fnv1a64: rds0_fnv1a64(drainage_words),
        minimum_elevation: finite[0],
        median_elevation: at(0.5),
        p95_elevation: at(0.95),
        maximum_elevation: *finite.last().unwrap(),
        positive_elevation_cell_fraction: values.iter().filter(|&&value| value > 0.0).count()
            as f64
            / values.len() as f64,
        ocean_cell_fraction: surface
            .hydrology
            .is_ocean
            .iter()
            .filter(|&&ocean| ocean)
            .count() as f64
            / values.len() as f64,
        lake_count: surface.hydrology.water_bodies.len(),
    }
}

#[cfg(feature = "research-landscape")]
fn rds0_surface_delta(
    control: &[f32],
    candidate: &[f32],
    support: &[bool],
    control_drainage: &[Option<usize>],
    candidate_drainage: &[Option<usize>],
) -> Rds0SurfaceDelta {
    assert_eq!(control.len(), candidate.len());
    let mut absolute_sum = 0.0_f64;
    let mut square_sum = 0.0_f64;
    let mut maximum = 0.0_f32;
    let mut positive = 0_usize;
    let mut support_cells = 0_usize;
    let mut changed_receivers = 0_usize;
    let mut support_absolute_sum = 0.0_f64;
    let mut support_square_sum = 0.0_f64;
    let mut support_maximum = 0.0_f32;
    for (cell, (&a, &b)) in control.iter().zip(candidate).enumerate() {
        let delta = b - a;
        let absolute = delta.abs();
        absolute_sum += f64::from(absolute);
        square_sum += f64::from(delta) * f64::from(delta);
        maximum = maximum.max(absolute);
        positive += usize::from(delta > 0.0);
        if support[cell] {
            support_cells += 1;
            changed_receivers += usize::from(control_drainage[cell] != candidate_drainage[cell]);
            support_absolute_sum += f64::from(absolute);
            support_square_sum += f64::from(delta) * f64::from(delta);
            support_maximum = support_maximum.max(absolute);
        }
    }
    Rds0SurfaceDelta {
        mean_absolute_elevation_delta: absolute_sum / control.len() as f64,
        root_mean_square_elevation_delta: (square_sum / control.len() as f64).sqrt(),
        maximum_absolute_elevation_delta: maximum,
        positive_delta_cell_fraction: positive as f64 / control.len() as f64,
        support_mean_absolute_elevation_delta: support_absolute_sum / support_cells.max(1) as f64,
        support_root_mean_square_elevation_delta: (support_square_sum
            / support_cells.max(1) as f64)
            .sqrt(),
        support_maximum_absolute_elevation_delta: support_maximum,
        support_drainage_receiver_changed_fraction: changed_receivers as f64
            / support_cells.max(1) as f64,
    }
}

#[cfg(feature = "research-landscape")]
fn rds0_local_surface_summary(
    tessellation: &Tessellation,
    support: &[bool],
    surface: &FineSurface,
) -> Rds0LocalSurfaceSummary {
    let areas = tessellation.cell_areas_ref();
    let elevation = &surface.elevation.values;
    let support_area: f64 = support
        .iter()
        .zip(areas)
        .filter_map(|(&active, &area)| active.then_some(f64::from(area)))
        .sum();
    let weighted_sum: f64 = support
        .iter()
        .zip(areas)
        .zip(elevation)
        .filter_map(|((&active, &area), &z)| active.then_some(f64::from(area) * f64::from(z)))
        .sum();
    let mean = weighted_sum / support_area.max(f64::EPSILON);
    let variance: f64 = support
        .iter()
        .zip(areas)
        .zip(elevation)
        .filter_map(|((&active, &area), &z)| {
            active.then_some(f64::from(area) * (f64::from(z) - mean).powi(2))
        })
        .sum::<f64>()
        / support_area.max(f64::EPSILON);
    let mut slopes = Vec::new();
    let mut maximum = f32::NEG_INFINITY;
    for cell in 0..tessellation.num_cells() {
        if !support[cell] {
            continue;
        }
        maximum = maximum.max(elevation[cell]);
        let center = tessellation.cell_center(cell);
        let downhill = tessellation
            .neighbors(cell)
            .iter()
            .map(|&neighbor| {
                let distance = center
                    .dot(tessellation.cell_center(neighbor))
                    .clamp(-1.0, 1.0)
                    .acos()
                    .max(f32::EPSILON);
                ((elevation[cell] - elevation[neighbor]) / distance).max(0.0)
            })
            .fold(0.0_f32, f32::max);
        slopes.push(downhill);
    }
    assert!(
        !slopes.is_empty(),
        "RDS0 target-land union support is empty"
    );
    slopes.sort_unstable_by(f32::total_cmp);
    let slope_at = |fraction: f64| slopes[((slopes.len() - 1) as f64 * fraction).round() as usize];
    let cell_count = slopes.len();
    Rds0LocalSurfaceSummary {
        cell_count,
        cell_fraction: cell_count as f64 / tessellation.num_cells() as f64,
        spherical_area_fraction: support_area / (4.0 * std::f64::consts::PI),
        area_weighted_mean_elevation: mean,
        area_weighted_elevation_std_dev: variance.sqrt(),
        maximum_elevation: maximum,
        median_max_downhill_slope: slope_at(0.5),
        p95_max_downhill_slope: slope_at(0.95),
    }
}

#[cfg(feature = "research-landscape")]
fn rds0_capture_views(opts: &SweepOptions, parent_centers: [Vec3; 2]) -> Vec<CaptureView> {
    let aspect = opts.width as f32 / opts.height as f32;
    let mut camera = OrbitCamera::new();
    camera.yaw = opts.yaw_deg.to_radians();
    camera.pitch = opts.pitch_deg.to_radians();
    camera.distance = opts.distance;
    camera.aspect = aspect;
    let overview_eye = camera.eye_position();
    let mut views = vec![CaptureView {
        view_proj: camera.view_projection(),
        eye: overview_eye,
        label: "globe".to_string(),
        sidecar: ViewRecord {
            id: "globe".to_string(),
            kind: "overview",
            target: None,
            camera: CameraRecord {
                eye_xyz: vec3_array(overview_eye),
                aim_xyz: [0.0; 3],
                up_xyz: [0.0, 1.0, 0.0],
                vertical_fov_deg: 45.0,
                aspect,
                near: 0.01,
                far: 10.0,
                target_altitude: None,
                orbit_yaw_deg: Some(opts.yaw_deg),
                orbit_pitch_deg: Some(opts.pitch_deg),
                orbit_distance: Some(opts.distance),
            },
        },
    }];
    for (index, parent_center) in parent_centers.into_iter().enumerate() {
        let center = parent_center.normalize();
        let target = SweepTarget {
            id: format!("selected-parent-q{}", [33, 67][index]),
            latitude_deg: center.y.clamp(-1.0, 1.0).asin().to_degrees(),
            longitude_deg: center.z.atan2(center.x).to_degrees(),
        };
        let (regional_vp, regional_eye) = target_camera(center, aspect, opts.zoom_alt);
        views.push(CaptureView {
            view_proj: regional_vp,
            eye: regional_eye,
            label: target.id.clone(),
            sidecar: ViewRecord {
                id: target.id.clone(),
                kind: "selected-parent-length-quantile",
                target: Some(target),
                camera: CameraRecord {
                    eye_xyz: vec3_array(regional_eye),
                    aim_xyz: vec3_array(center * 1.08),
                    up_xyz: vec3_array(center),
                    vertical_fov_deg: 45.0,
                    aspect,
                    near: 0.01,
                    far: 10.0,
                    target_altitude: Some(opts.zoom_alt),
                    orbit_yaw_deg: None,
                    orbit_pitch_deg: None,
                    orbit_distance: None,
                },
            },
        });
    }
    views
}

#[cfg(feature = "research-landscape")]
fn b0_scaffold_colors_and_audit(
    result: &B0DrainageDualResultV0,
    surface: &FineSurface,
    support: &[bool],
) -> (Vec<Vec3>, B0ScaffoldFinalAuditV0) {
    let n = result.scaffold_receiver.len();
    assert_eq!(surface.hydrology.drainage_dir.len(), n);
    assert_eq!(support.len(), n);

    let mut terminal = vec![usize::MAX; n];
    for start in 0..n {
        if terminal[start] != usize::MAX {
            continue;
        }
        let mut path = Vec::new();
        let mut cell = start;
        while terminal[cell] == usize::MAX && result.scaffold_receiver[cell] != cell {
            path.push(cell);
            cell = result.scaffold_receiver[cell];
        }
        let sink = if terminal[cell] != usize::MAX {
            terminal[cell]
        } else {
            terminal[cell] = cell;
            cell
        };
        for member in path {
            terminal[member] = sink;
        }
    }

    let agrees = |cell: usize| {
        let scaffold = result.scaffold_receiver[cell];
        surface.hydrology.drainage_dir[cell] == Some(scaffold)
            || (scaffold == cell && surface.hydrology.drainage_dir[cell].is_none())
    };
    let mut agreement = 0usize;
    let mut support_agreement = 0usize;
    let mut support_count = 0usize;
    let mut channel_agreement = 0usize;
    let mut channel_count = 0usize;
    let mut colors = Vec::with_capacity(n);
    for cell in 0..n {
        let same = agrees(cell);
        agreement += usize::from(same);
        if support[cell] {
            support_count += 1;
            support_agreement += usize::from(same);
        }
        if result.promoted_channel[cell] {
            channel_count += 1;
            channel_agreement += usize::from(same);
        }

        let hash = terminal[cell].wrapping_mul(0x9e37_79b1);
        let mut color = Vec3::new(
            0.12 + 0.18 * ((hash & 255) as f32 / 255.0),
            0.12 + 0.18 * (((hash >> 8) & 255) as f32 / 255.0),
            0.12 + 0.18 * (((hash >> 16) & 255) as f32 / 255.0),
        );
        if result.promoted_channel[cell] {
            let order = result.scaffold_strahler_order[cell].min(5) as f32;
            color = Vec3::new(0.05, 0.55 + 0.08 * order, 0.9);
        }
        if !same {
            color = Vec3::new(0.95, 0.08, 0.04);
        }
        colors.push(color);
    }
    let terminal_count = terminal.iter().copied().collect::<BTreeSet<_>>().len();
    let audit = B0ScaffoldFinalAuditV0 {
        scaffold_receiver_fnv1a64: rds0_fnv1a64(
            result.scaffold_receiver.iter().map(|&cell| cell as u64),
        ),
        promoted_channel_fnv1a64: rds0_fnv1a64(
            result
                .promoted_channel
                .iter()
                .map(|&value| u64::from(value)),
        ),
        strahler_order_fnv1a64: rds0_fnv1a64(
            result
                .scaffold_strahler_order
                .iter()
                .map(|&order| u64::from(order)),
        ),
        scaffold_terminal_count: terminal_count,
        final_receiver_agreement_fraction: agreement as f64 / n.max(1) as f64,
        support_receiver_agreement_fraction: support_agreement as f64 / support_count.max(1) as f64,
        promoted_channel_receiver_agreement_fraction: channel_agreement as f64
            / channel_count.max(1) as f64,
        mismatch_cell_count: n - agreement,
        montage_filename: "b0_scaffold_montage.png",
        montage_column_order: ["selected-parent-q33", "selected-parent-q67"],
    };
    (colors, audit)
}

#[cfg(feature = "research-landscape")]
#[allow(clippy::too_many_arguments)]
fn render_b0_scaffold(
    opts: &SweepOptions,
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_texture: &wgpu::Texture,
    color_view: &wgpu::TextureView,
    views: &[CaptureView],
    world: &mut World,
    colors: &[Vec3],
) {
    let mut buffers = generate_world_buffers(&gpu.device, &gpu.queue, world);
    regenerate_diagnostic_surface_colors(&gpu.queue, world, &mut buffers, colors)
        .unwrap_or_else(|error| panic!("B0 scaffold colors: {error}"));
    let montage_width = opts.width * (views.len() as u32 - 1);
    let mut montage = vec![0; (montage_width * opts.height * 4) as usize];
    for (column, view) in views.iter().skip(1).enumerate() {
        render_relief(
            gpu,
            renderer,
            color_view,
            &buffers,
            view.view_proj,
            view.eye,
            RiverMode::Off,
            PHYSICAL_RELIEF_SCALE,
            1.0,
        );
        let rgba = read_back_rgba(gpu, color_texture, opts.width, opts.height);
        write_png(
            &opts.out_dir.join(format!("b0_scaffold_{}.png", view.label)),
            &rgba,
            opts.width,
            opts.height,
        );
        blit_tile(
            &mut montage,
            montage_width,
            &rgba,
            opts.width,
            opts.height,
            column as u32,
            0,
        );
    }
    write_png(
        &opts.out_dir.join("b0_scaffold_montage.png"),
        &montage,
        montage_width,
        opts.height,
    );
}

#[cfg(feature = "research-landscape")]
#[allow(clippy::too_many_arguments)]
fn render_rds0_surface(
    opts: &SweepOptions,
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_texture: &wgpu::Texture,
    color_view: &wgpu::TextureView,
    views: &[CaptureView],
    montage: &mut [u8],
    montage_width: u32,
    relationship_montage: &mut [u8],
    relationship_montage_width: u32,
    world: &mut World,
    surface: FineSurface,
    relationship: &RdsRelationshipAnalysisV0,
    arm: &'static str,
    arm_index: usize,
    retain_surface: bool,
) -> (Vec<Rds0RenderRecord>, Vec<Rds0RenderRecord>) {
    world.fine.as_mut().expect("RDS0 fine world").eroded = Some(surface);
    world.set_view_stage(4);
    let mut buffers = generate_world_buffers(&gpu.device, &gpu.queue, world);
    let presentations = [
        ("physical", PHYSICAL_RELIEF_SCALE),
        ("authentic", RELIEF_SCALE),
    ];
    let mut records = Vec::with_capacity(presentations.len());
    for (presentation_index, (presentation, relief_scale)) in presentations.into_iter().enumerate()
    {
        let montage_row = presentation_index * RDS0_TERRAIN_ARM_COUNT + arm_index;
        let mut filenames = Vec::with_capacity(views.len());
        for (view_index, view) in views.iter().enumerate() {
            render_relief(
                gpu,
                renderer,
                color_view,
                &buffers,
                view.view_proj,
                view.eye,
                RiverMode::Major,
                relief_scale,
                1.0,
            );
            let rgba = read_back_rgba(gpu, color_texture, opts.width, opts.height);
            let filename = format!("{arm}_{presentation}_{}.png", view.label);
            write_png(
                &opts.out_dir.join(&filename),
                &rgba,
                opts.width,
                opts.height,
            );
            blit_tile(
                montage,
                montage_width,
                &rgba,
                opts.width,
                opts.height,
                view_index as u32,
                montage_row as u32,
            );
            filenames.push(filename);
        }
        records.push(Rds0RenderRecord {
            arm,
            presentation,
            relief_scale,
            image_filenames: filenames,
        });
    }
    let relationship_presentations = [
        (
            "source-topology",
            relationship.source_topology_colors.as_slice(),
        ),
        (
            "catchment-divide-basin",
            relationship.catchment_divide_basin_colors.as_slice(),
        ),
    ];
    let mut relationship_records = Vec::with_capacity(relationship_presentations.len());
    for (overlay_index, (presentation, colors)) in
        relationship_presentations.into_iter().enumerate()
    {
        regenerate_diagnostic_surface_colors(&gpu.queue, world, &mut buffers, colors)
            .unwrap_or_else(|error| panic!("RDS0 relationship colors: {error}"));
        let mut filenames = Vec::with_capacity(views.len().saturating_sub(1));
        for (regional_index, view) in views.iter().skip(1).enumerate() {
            render_relief(
                gpu,
                renderer,
                color_view,
                &buffers,
                view.view_proj,
                view.eye,
                RiverMode::Major,
                PHYSICAL_RELIEF_SCALE,
                1.0,
            );
            let rgba = read_back_rgba(gpu, color_texture, opts.width, opts.height);
            let filename = format!("{arm}_relationship-{presentation}_{}.png", view.label);
            write_png(
                &opts.out_dir.join(&filename),
                &rgba,
                opts.width,
                opts.height,
            );
            blit_tile(
                relationship_montage,
                relationship_montage_width,
                &rgba,
                opts.width,
                opts.height,
                (overlay_index * (views.len() - 1) + regional_index) as u32,
                arm_index as u32,
            );
            filenames.push(filename);
        }
        relationship_records.push(Rds0RenderRecord {
            arm,
            presentation,
            relief_scale: PHYSICAL_RELIEF_SCALE,
            image_filenames: filenames,
        });
    }
    drop(buffers);
    if !retain_surface {
        world.fine.as_mut().unwrap().eroded.take();
    }
    (records, relationship_records)
}

/// Fixed RDS0 terrain packet: the source-only four-frame program is transferred
/// conservatively to one ordinary fine process mesh, then compared against its
/// static nearest-source control under the same Legacy builder budget.
#[cfg(feature = "research-landscape")]
fn run_rds0_terrain_packet(opts: &SweepOptions) {
    assert_eq!(opts.seed, RDS0_TERRAIN_SEED);
    assert_eq!(opts.cells, RDS0_TERRAIN_COARSE_CELLS);
    assert_eq!(opts.fine_scale, 1.0);
    assert_eq!(opts.fine_max, RDS0_TERRAIN_FINE_CAP);
    assert_eq!(opts.target_stage, 4);
    assert_eq!(opts.voronoi_backend, VoronoiBackend::ConvexHull);
    assert_eq!(opts.orogen_model, OrogenModel::Legacy);
    assert_eq!(opts.fine_cache, FineCacheMode::Disabled);
    assert!(opts.targets.is_empty(), "RDS0 cameras are source-derived");
    assert_eq!(opts.display_subdivision_levels, 0);

    let started = Instant::now();
    let mut world = create_world_with_orogen_model(
        opts.seed,
        opts.cells,
        VoronoiBackend::ConvexHull,
        FineCacheMode::Disabled,
        OrogenModel::Legacy,
    );
    let neutral = ErosionOverrides {
        finite_age_uplift: true,
        ..ErosionOverrides::default()
    };
    neutral.apply(&mut world);
    advance_to_stage_2(&mut world);
    advance_to_stage_3_with_cap(&mut world, RDS0_TERRAIN_FINE_CAP);

    let boundaries = collect_plate_boundaries(
        &world.tessellation,
        world.plates.as_ref().expect("plates generated"),
        world.crust.as_ref().expect("crust generated"),
        world.dynamics.as_ref().expect("dynamics generated"),
    );
    let fronts = collect_convergent_fronts(
        &world.tessellation,
        &boundaries,
        world.tectonic_history.as_ref().expect("history generated"),
    )
    .expect("RDS0 exact fronts");
    let program: RegionalDeformationProgramV0 =
        build_regional_deformation_rds0_v0(&fronts, RDS0_TERRAIN_EPISODE)
            .expect("RDS0 source program");
    assert_eq!(program.frames.len(), RDS0_FRAME_COUNT);
    assert!(
        program.omissions.is_empty(),
        "RDS0 program omissions are a hard gate"
    );
    let parent_source_edges: Vec<_> = program
        .parent_source_edges
        .iter()
        .map(|id| {
            fronts
                .edges
                .iter()
                .find(|edge| edge.id == *id)
                .expect("selected parent source edge retained in exact fronts")
        })
        .collect();
    let parent_total_length: f64 = parent_source_edges
        .iter()
        .map(|edge| f64::from(edge.length_km))
        .sum();
    let parent_center_at = |fraction: f64| {
        let target = parent_total_length * fraction;
        let mut cumulative = 0.0;
        for edge in &parent_source_edges {
            cumulative += f64::from(edge.length_km);
            if cumulative >= target {
                return edge.midpoint;
            }
        }
        parent_source_edges.last().unwrap().midpoint
    };
    let parent_centers = [parent_center_at(1.0 / 3.0), parent_center_at(2.0 / 3.0)];
    let source_control = evaluate_regional_deformation_static_control_v0(
        &program,
        &fronts,
        &world.tessellation,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
    )
    .expect("RDS0 static source control");
    let source_frames: Vec<RegionalDeformationRasterV0> = (0..RDS0_FRAME_COUNT)
        .map(|frame| {
            evaluate_regional_deformation_frame_v0(
                &program,
                frame,
                &fronts,
                &world.tessellation,
                world.plates.as_ref().unwrap(),
                world.crust.as_ref().unwrap(),
            )
            .expect("RDS0 source frame")
        })
        .collect();
    assert!(source_control.omissions.is_empty());
    assert!(source_frames.iter().all(|frame| frame.omissions.is_empty()));

    let transfer_preflight = rds0_transfer_preflight(
        &world.tessellation,
        &world
            .fine
            .as_ref()
            .expect("RDS0 Stage-3 fine base")
            .base
            .coarse_cell,
        &source_control,
        &source_frames,
    );
    println!(
        "RDS0 transfer preflight: {}/{} coarse owners represented ({} absent)",
        transfer_preflight.represented_coarse_owner_count,
        transfer_preflight.coarse_owner_count,
        transfer_preflight.unrepresented_coarse_owner_count,
    );
    for raster in &transfer_preflight.rasters {
        println!(
            "  {}: {} active absent owners, {:.9} km²/Myr unrepresented, sample {:?}",
            raster.field,
            raster.active_unrepresented_owner_count,
            raster.requested_unrepresented_opportunity_km2_per_myr,
            raster.owner_id_sample,
        );
    }
    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|error| panic!("create {}: {error}", opts.out_dir.display()));
    let file = std::fs::File::create(opts.out_dir.join("rds0-transfer-preflight.json"))
        .expect("create rds0-transfer-preflight.json");
    serde_json::to_writer_pretty(BufWriter::new(file), &transfer_preflight)
        .expect("write rds0-transfer-preflight.json");
    if !transfer_preflight.center_owner_transfer_feasible {
        println!(
            "  center-owner transfer is underresolved; continuing with strict exact polygon overlap"
        );
    }

    let (mut fine_control, mut fine_frames, overlap_map_audit, shared_input_hashes) = {
        let fine = world.fine.as_ref().expect("RDS0 Stage-3 fine base");
        let union_rasters: Vec<&RegionalDeformationRasterV0> = std::iter::once(&source_control)
            .chain(source_frames.iter())
            .collect();
        let overlap_map = build_regional_deformation_overlap_map_v0(
            &world.tessellation,
            &fine.base,
            &union_rasters,
        )
        .expect("RDS0 union exact-overlap map");
        let control = transfer_regional_deformation_raster_with_overlap_v0(
            &world.tessellation,
            &fine.base,
            &overlap_map,
            &source_control,
        )
        .expect("RDS0 exact-overlap control transfer");
        let frames: Vec<RegionalDeformationRasterV0> = source_frames
            .iter()
            .map(|frame| {
                transfer_regional_deformation_raster_with_overlap_v0(
                    &world.tessellation,
                    &fine.base,
                    &overlap_map,
                    frame,
                )
                .expect("RDS0 exact-overlap frame transfer")
            })
            .collect();
        let overlap_audit = serde_json::json!({
            "coarse_cell_count": overlap_map.coarse_cell_count,
            "fine_cell_count": overlap_map.fine_cell_count,
            "donor_count": overlap_map.donor_count,
            "pair_count": overlap_map.pair_count,
            "maximum_geometric_coverage_relative_error": overlap_map.maximum_geometric_coverage_relative_error,
            "minimum_recipient_fraction": overlap_map.minimum_recipient_fraction,
            "donor_audits": overlap_map.donor_audits,
        });
        let tessellation = &fine.base.tessellation;
        let hashes = serde_json::json!({
            "fine_centers_fnv1a64": rds0_fnv1a64(
                tessellation.voronoi.generators.iter().flat_map(|center| [
                    u64::from(center.x.to_bits()), u64::from(center.y.to_bits()),
                    u64::from(center.z.to_bits()),
                ])
            ),
            "fine_neighbor_topology_fnv1a64": rds0_fnv1a64(
                (0..tessellation.num_cells()).flat_map(|cell| {
                    std::iter::once(cell as u64)
                        .chain(tessellation.neighbors(cell).iter().map(|&neighbor| neighbor as u64))
                        .chain(std::iter::once(u64::MAX))
                })
            ),
            "fine_to_coarse_owner_fnv1a64": rds0_fnv1a64(
                fine.base.coarse_cell.iter().map(|&cell| cell as u64)
            ),
            "demoted_base_elevation_fnv1a64": rds0_fnv1a64(
                fine.base.base_elevation.iter().map(|value| u64::from(value.to_bits()))
            ),
            "coarse_target_elevation_fnv1a64": rds0_fnv1a64(
                fine.base.coarse_base_elevation.iter().map(|value| u64::from(value.to_bits()))
            ),
            "initial_precipitation_fnv1a64": rds0_fnv1a64(
                fine.base.fields.precipitation.iter().map(|value| u64::from(value.to_bits()))
            ),
            "initial_temperature_fnv1a64": rds0_fnv1a64(
                fine.base.fields.temperature.iter().map(|value| u64::from(value.to_bits()))
            ),
        });
        (control, frames, overlap_audit, hashes)
    };
    assert!(fine_control.omissions.is_empty());
    assert!(fine_frames.iter().all(|frame| frame.omissions.is_empty()));
    let target_land_union_support: Vec<bool> = world
        .fine
        .as_ref()
        .unwrap()
        .base
        .coarse_base_elevation
        .iter()
        .enumerate()
        .map(|(cell, &target)| {
            target > 0.0
                && (fine_control.rate_density_per_myr[cell] > 0.0
                    || fine_frames
                        .iter()
                        .any(|frame| frame.rate_density_per_myr[cell] > 0.0))
        })
        .collect();
    let source_ledgers: Vec<RegionalDeformationRasterLedgerV0> =
        std::iter::once(source_control.ledger.clone())
            .chain(source_frames.iter().map(|frame| frame.ledger.clone()))
            .collect();
    let transfer_ledgers: Vec<RegionalDeformationRasterLedgerV0> =
        std::iter::once(fine_control.ledger.clone())
            .chain(fine_frames.iter().map(|frame| frame.ledger.clone()))
            .collect();
    let control_source_relationship = {
        let frames = [&fine_control; RDS0_FRAME_COUNT];
        compress_rds0_schedule_v0(&frames).expect("RDS0 control relationship source")
    };
    let candidate_source_relationship = {
        let frames: [&RegionalDeformationRasterV0; RDS0_FRAME_COUNT] =
            std::array::from_fn(|frame| &fine_frames[frame]);
        compress_rds0_schedule_v0(&frames).expect("RDS0 candidate relationship source")
    };
    // Erosion consumes density only. Release source meshes and fine transfer
    // provenance before either long Stage-4 arm so this diagnostic does not
    // recreate the high peak-RAM behavior it is meant to replace.
    drop(source_control);
    drop(source_frames);
    for raster in std::iter::once(&mut fine_control).chain(fine_frames.iter_mut()) {
        raster.active_support_fraction.clear();
        raster.axial_fabric.clear();
        raster.provenance.clear();
        raster.omissions.clear();
    }

    let params = world.erosion_params;
    let lookback_myr = f64::from(
        world
            .tectonic_history
            .as_ref()
            .expect("history generated")
            .lookback_myr,
    );
    let (control_surface, control_audit): (FineSurface, LegacyBudgetOpportunityAuditV0) = {
        let fine = world.fine.as_ref().unwrap();
        let control_schedule = [&fine_control; RDS0_FRAME_COUNT];
        FineSurface::generate_regional_deformation_v0(
            opts.seed,
            &fine.base,
            &fine.pre.hydrology,
            params,
            control_schedule,
            program.parent_duration_myr,
            lookback_myr,
        )
        .expect("RDS0 control terrain")
    };
    let control_summary = rds0_surface_summary(&control_surface);
    let control_local_summary = rds0_local_surface_summary(
        &world.fine.as_ref().unwrap().base.tessellation,
        &target_land_union_support,
        &control_surface,
    );
    let control_relationship = analyze_rds0_relationships_v0(
        &world.fine.as_ref().unwrap().base.tessellation,
        &control_surface,
        &target_land_union_support,
        &control_source_relationship,
    )
    .expect("RDS0 control relationship readout");
    let control_relationship_summary = control_relationship.summary.clone();
    let control_elevation = control_surface.elevation.values.clone();
    let control_drainage = control_surface.hydrology.drainage_dir.clone();

    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|error| panic!("create {}: {error}", opts.out_dir.display()));
    let gpu = pollster::block_on(GpuContext::new_headless(opts.width, opts.height));
    let mut renderer = Renderer::new(&gpu, &Uniforms::new(Mat4::IDENTITY, Vec3::ZERO, Vec3::Y));
    let color_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rds0_terrain_color"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&Default::default());
    let views = rds0_capture_views(opts, parent_centers);
    let montage_width = opts.width * views.len() as u32;
    let montage_height = opts.height * (2 * RDS0_TERRAIN_ARM_COUNT) as u32;
    let mut montage = vec![0; (montage_width * montage_height * 4) as usize];
    let relationship_montage_width = opts.width * (views.len() as u32 - 1) * 2;
    let relationship_montage_height = opts.height * RDS0_TERRAIN_ARM_COUNT as u32;
    let mut relationship_montage =
        vec![0; (relationship_montage_width * relationship_montage_height * 4) as usize];
    let (mut render_records, mut relationship_render_records) = render_rds0_surface(
        opts,
        &gpu,
        &mut renderer,
        &color_texture,
        &color_view,
        &views,
        &mut montage,
        montage_width,
        &mut relationship_montage,
        relationship_montage_width,
        &mut world,
        control_surface,
        &control_relationship,
        "control",
        0,
        false,
    );
    drop(control_relationship);

    let (candidate_surface, candidate_audit): (FineSurface, LegacyBudgetOpportunityAuditV0) = {
        let fine = world.fine.as_ref().unwrap();
        let candidate_schedule: [&RegionalDeformationRasterV0; RDS0_FRAME_COUNT] =
            std::array::from_fn(|frame| &fine_frames[frame]);
        FineSurface::generate_regional_deformation_v0(
            opts.seed,
            &fine.base,
            &fine.pre.hydrology,
            params,
            candidate_schedule,
            program.parent_duration_myr,
            lookback_myr,
        )
        .expect("RDS0 candidate terrain")
    };
    let candidate_summary = rds0_surface_summary(&candidate_surface);
    let candidate_local_summary = rds0_local_surface_summary(
        &world.fine.as_ref().unwrap().base.tessellation,
        &target_land_union_support,
        &candidate_surface,
    );
    let surface_delta = rds0_surface_delta(
        &control_elevation,
        &candidate_surface.elevation.values,
        &target_land_union_support,
        &control_drainage,
        &candidate_surface.hydrology.drainage_dir,
    );
    let candidate_relationship = analyze_rds0_relationships_v0(
        &world.fine.as_ref().unwrap().base.tessellation,
        &candidate_surface,
        &target_land_union_support,
        &candidate_source_relationship,
    )
    .expect("RDS0 candidate relationship readout");
    let candidate_relationship_summary = candidate_relationship.summary.clone();
    let candidate_elevation = candidate_surface.elevation.values.clone();
    let candidate_drainage = candidate_surface.hydrology.drainage_dir.clone();
    let (candidate_render_records, candidate_relationship_render_records) = render_rds0_surface(
        opts,
        &gpu,
        &mut renderer,
        &color_texture,
        &color_view,
        &views,
        &mut montage,
        montage_width,
        &mut relationship_montage,
        relationship_montage_width,
        &mut world,
        candidate_surface,
        &candidate_relationship,
        "candidate",
        1,
        false,
    );
    render_records.extend(candidate_render_records);
    relationship_render_records.extend(candidate_relationship_render_records);
    drop(candidate_relationship);
    drop(fine_control);
    drop(fine_frames);

    let (b0_surface, b0_result) = {
        let fine = world.fine.as_ref().unwrap();
        FineSurface::generate_channel_hillslope_dual_b0(
            &fine.base,
            &fine.pre.hydrology,
            params,
            &candidate_source_relationship.mean_source_density_per_myr,
            &candidate_source_relationship.axial_fabric,
        )
        .expect("RDS0 B0 channel/hillslope-dual terrain")
    };
    let b0_summary = rds0_surface_summary(&b0_surface);
    let b0_local_summary = rds0_local_surface_summary(
        &world.fine.as_ref().unwrap().base.tessellation,
        &target_land_union_support,
        &b0_surface,
    );
    let b0_minus_control = rds0_surface_delta(
        &control_elevation,
        &b0_surface.elevation.values,
        &target_land_union_support,
        &control_drainage,
        &b0_surface.hydrology.drainage_dir,
    );
    let b0_minus_candidate = rds0_surface_delta(
        &candidate_elevation,
        &b0_surface.elevation.values,
        &target_land_union_support,
        &candidate_drainage,
        &b0_surface.hydrology.drainage_dir,
    );
    let b0_relationship = analyze_rds0_relationships_v0(
        &world.fine.as_ref().unwrap().base.tessellation,
        &b0_surface,
        &target_land_union_support,
        &candidate_source_relationship,
    )
    .expect("RDS0 B0 relationship readout");
    let b0_relationship_summary = b0_relationship.summary.clone();
    let (b0_scaffold_colors, b0_scaffold_final_audit) =
        b0_scaffold_colors_and_audit(&b0_result, &b0_surface, &target_land_union_support);
    let (b0_render_records, b0_relationship_render_records) = render_rds0_surface(
        opts,
        &gpu,
        &mut renderer,
        &color_texture,
        &color_view,
        &views,
        &mut montage,
        montage_width,
        &mut relationship_montage,
        relationship_montage_width,
        &mut world,
        b0_surface,
        &b0_relationship,
        "b0",
        2,
        true,
    );
    render_records.extend(b0_render_records);
    relationship_render_records.extend(b0_relationship_render_records);
    render_b0_scaffold(
        opts,
        &gpu,
        &mut renderer,
        &color_texture,
        &color_view,
        &views,
        &mut world,
        &b0_scaffold_colors,
    );
    write_png(
        &opts.out_dir.join("montage.png"),
        &montage,
        montage_width,
        montage_height,
    );
    write_png(
        &opts.out_dir.join("relationships_montage.png"),
        &relationship_montage,
        relationship_montage_width,
        relationship_montage_height,
    );
    drop(b0_relationship);

    let sidecar = serde_json::json!({
        "schema": "hex3.rds0-terrain.v1",
        "status": "research-only fixed-budget causal morphology discriminator with B0 channel/hillslope upper bound; not promoted product terrain",
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "config": {
            "seed": RDS0_TERRAIN_SEED,
            "requested_coarse_cells": RDS0_TERRAIN_COARSE_CELLS,
            "fine_cell_cap": RDS0_TERRAIN_FINE_CAP,
            "episode_id": RDS0_TERRAIN_EPISODE,
            "stage": 4,
            "voronoi_backend": "convex-hull",
            "orogen_model": "legacy",
            "fine_cache": "disabled",
            "scientific_cli_controls_consumed": 0,
        },
        "world_manifest": world.manifest(),
        "transfer_preflight": transfer_preflight,
        "overlap_map_audit": overlap_map_audit,
        "shared_input_hashes": shared_input_hashes,
        "source_program": program,
        "source_mesh_ledgers": source_ledgers,
        "fine_transfer_ledgers": transfer_ledgers,
        "adapter_ledgers": {
            "control": control_audit,
            "candidate": candidate_audit,
            "b0": null,
        },
        "surfaces": {
            "control": { "global": control_summary, "target_land_union_support": control_local_summary },
            "candidate": { "global": candidate_summary, "target_land_union_support": candidate_local_summary },
            "candidate_minus_control": surface_delta,
            "b0": { "global": b0_summary, "target_land_union_support": b0_local_summary },
            "b0_minus_control": b0_minus_control,
            "b0_minus_candidate": b0_minus_candidate,
        },
        "b0": {
            "schema": "hex3.channel-hillslope-dual-b0.v0",
            "semantics": "all-cell product receiver scaffold; sparse opportunity/fabric-conditioned channel graph; reduced steady slope-area profiles integrated from real base levels; affine finite-volume non-channel reconstruction; one solved amplitude under the removed-Legacy solid/relief bounds",
            "profile": {
                "unit_slope_law": "S* = max(epsilon, (normalized opportunity / represented drainage area^m)^(1/n))",
                "stream_power_m": params.m,
                "stream_power_n": params.n,
                "amplitude": "one globally solved scalar; maximum value satisfying both positive-solid and maximum-relief bounds",
            },
            "audit": b0_result.audit,
            "scaffold_vs_final": b0_scaffold_final_audit,
        },
        "relationships": {
            "control": control_relationship_summary,
            "candidate": candidate_relationship_summary,
            "b0": b0_relationship_summary,
            "render_rows": relationship_render_records,
            "montage_filename": "relationships_montage.png",
            "montage_row_order": ["control", "candidate", "b0"],
            "montage_column_order": [
                "source topology q33", "source topology q67",
                "terminal-mouth catchments / boundary proxies / depressions q33",
                "terminal-mouth catchments / boundary proxies / depressions q67"
            ],
        },
        "cameras": views.iter().map(|view| &view.sidecar).collect::<Vec<_>>(),
        "render": {
            "rows": render_records,
            "montage_filename": "montage.png",
            "montage_row_order": ["physical control", "physical candidate", "physical b0", "authentic control", "authentic candidate", "authentic b0"],
            "palette": "ordinary Terrain palette",
            "rivers": "major, identical physical catchment selection and width scale 1.0",
        },
        "declared_confounds": [
            "the full global demoted-Legacy builder budget is concentrated onto one selected parent in both arms; this is an intentional fixed-budget morphology subsidy, not a physical amplitude calibration",
            "the shared minimum builder floor can dominate the shaped excess and is not RDS uplift",
            "the adapter rejects opportunity outside the fixed target-land gate and renormalizes surviving opportunity; the adapter ledgers report the resulting budget",
            "the moving four-frame source does not advect previously uplifted material horizontally",
            "RDS0 consumes no lithospheric inheritance relationships by construction",
            "the ordinary center-owner map omits some active coarse donors and is retained only as preflight evidence; terrain forcing uses one strict exact polygon-overlap map shared by control and all four frames",
            "physical and Authentic rows change renderer displacement only; both show identical semantic terrain state",
            "relationship colors are renderer-only physical-relief diagnostics over the same terrain state; white catchment boundaries are terminal-mouth graph-label proxies, not geomorphic divide claims",
            "rigorous saddle extraction is deliberately omitted from this RAM-bounded first readout because the existing exact implementation materializes a second full f64 spherical process-mesh graph",
            "B0 consumes time-integrated RDS opportunity and axial fabric, not chronological source motion or direct per-cell height",
            "B0 uses the removed Legacy terrain only for a global solid-volume and maximum-relief cap; unused budget is reported rather than concentrated into over-height peaks",
        ],
        "consumed_inheritance_relationship_ids": [],
    });
    let file = std::fs::File::create(opts.out_dir.join("rds0-terrain.json"))
        .expect("create rds0-terrain.json");
    serde_json::to_writer_pretty(BufWriter::new(file), &sidecar).expect("write rds0-terrain.json");
    println!("Done: RDS0 terrain packet -> {}", opts.out_dir.display());
}

/// Run the sweep: generate + render every knob combination to PNG tiles and a
/// stitched montage in `opts.out_dir`.
pub fn run_sweep(opts: SweepOptions) {
    if !opts.zoom_alt.is_finite() || opts.zoom_alt <= 0.0 {
        panic!("--sweep-zoom-alt must be finite and greater than zero");
    }
    assert!(
        opts.display_subdivision_levels <= 1,
        "--sweep-display-subdivision is a bounded 0/1 discriminator, not a display-LOD ladder"
    );
    let mut target_ids = std::collections::HashSet::new();
    for target in &opts.targets {
        if !target_ids.insert(target.id.as_str()) {
            panic!("duplicate --sweep-target id '{}'", target.id);
        }
    }
    if opts.stack.as_deref() == Some("rds0-terrain") {
        #[cfg(feature = "research-landscape")]
        {
            run_rds0_terrain_packet(&opts);
            return;
        }
        #[cfg(not(feature = "research-landscape"))]
        panic!("rds0-terrain requires --features research-landscape");
    }
    if opts.stack.as_deref() == Some("range-ancestry") {
        run_range_ancestry(&opts);
        return;
    }
    if opts.stack.as_deref() == Some("roof-compiler-counterfactual") {
        #[cfg(feature = "research-landscape")]
        {
            run_roof_compiler_counterfactual(&opts);
            return;
        }
        #[cfg(not(feature = "research-landscape"))]
        panic!("roof-compiler-counterfactual requires --features research-landscape");
    }
    if opts.stack.as_deref() == Some("water-geography") {
        run_water_geography_packet(&opts);
        return;
    }
    if opts.stack.as_deref() == Some("living-surface-preview") {
        run_living_surface_preview(&opts);
        return;
    }
    if opts.stack.as_deref() == Some("consequential-geography") {
        run_consequential_geography_packet(&opts);
        return;
    }
    if opts.stack.as_deref() == Some("world-readability-v0") {
        run_world_readability_packet(&opts);
        return;
    }
    // Validate knob names up front so a typo fails before any (slow) generation
    // (knob sweeps only; a stack preset uses fixed, pre-validated knobs).
    if opts.stack.is_none() {
        let mut probe = ErosionOverrides::default();
        apply_knob(&mut probe, &opts.knob1, opts.values1[0]).unwrap_or_else(|e| panic!("{e}"));
        if let Some(k2) = &opts.knob2 {
            apply_knob(&mut probe, k2, opts.values2[0]).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|e| panic!("create {}: {e}", opts.out_dir.display()));

    // Tile list: a cumulative stack preset, or the flattened knob grid (row-major:
    // knob2 outer, knob1 inner).
    let tiles: Vec<(ErosionOverrides, String, String, Vec<KnobValue>)> =
        if let Some(stack) = &opts.stack {
            build_stack_tiles(stack, &opts.base_erosion)
                .into_iter()
                .map(|(overrides, label, fname)| (overrides, label, fname, Vec::new()))
                .collect()
        } else {
            let rows: Vec<Option<f64>> = if opts.knob2.is_some() && !opts.values2.is_empty() {
                opts.values2.iter().map(|&v| Some(v)).collect()
            } else {
                vec![None]
            };
            let mut tiles = Vec::new();
            for row_val in &rows {
                for &v1 in &opts.values1 {
                    let mut overrides = opts.base_erosion;
                    apply_knob(&mut overrides, &opts.knob1, v1).unwrap();
                    let mut label = format!("{}={}", opts.knob1, fmt_value(v1));
                    let mut fname = format!("{}_{}", opts.knob1, fmt_value(v1));
                    let mut knob_values = vec![KnobValue {
                        name: opts.knob1.clone(),
                        value: v1,
                    }];
                    if let (Some(k2), Some(v2)) = (&opts.knob2, row_val) {
                        apply_knob(&mut overrides, k2, *v2).unwrap();
                        label = format!("{label}, {}={}", k2, fmt_value(*v2));
                        fname = format!("{fname}__{}_{}", k2, fmt_value(*v2));
                        knob_values.push(KnobValue {
                            name: k2.clone(),
                            value: *v2,
                        });
                    }
                    tiles.push((overrides, label, fname, knob_values));
                }
            }
            tiles
        };
    let n_tiles = tiles.len();

    let what = if let Some(stack) = &opts.stack {
        format!("stack '{stack}'")
    } else {
        format!(
            "{} ({} values){}",
            opts.knob1,
            opts.values1.len(),
            opts.knob2
                .as_ref()
                .map(|k| format!(" x {} ({} values)", k, opts.values2.len()))
                .unwrap_or_default(),
        )
    };
    println!(
        "Sweep: {} -> {} tiles x {} views at {}x{}, stage {}, seed {}",
        what,
        n_tiles,
        1 + if opts.targets.is_empty() {
            opts.zoom_views
        } else {
            opts.targets.len()
        },
        opts.width,
        opts.height,
        opts.target_stage,
        opts.seed,
    );

    // Headless GPU + renderer + offscreen color target (depth lives in Renderer).
    let gpu = pollster::block_on(GpuContext::new_headless(opts.width, opts.height));
    let init_uniforms = Uniforms::new(Mat4::IDENTITY, Vec3::ZERO, Vec3::Y);
    let mut renderer = Renderer::new(&gpu, &init_uniforms);

    let color_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sweep_color"),
        size: wgpu::Extent3d {
            width: opts.width,
            height: opts.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&Default::default());

    // Views are picked once from the first tile's terrain and reused for every
    // tile, so each montage column is the same region/angle across knob values
    // (the macro-geography is fixed by the seed; erosion knobs don't relocate it).
    // Montage rows = tiles, columns = views.
    let mut views: Vec<CaptureView> = Vec::new();
    let mut montage: Vec<u8> = Vec::new();
    let mut montage_w = 0u32;
    let mut tile_records = Vec::with_capacity(n_tiles);

    // Presentation sweeps are renderer-only: generate the terrain and GPU
    // buffers once so every row differs only in drawing policy.
    let render_only_presentation = opts.stack.is_none()
        && matches!(opts.knob1.as_str(), "relief_scale" | "river_width_scale")
        && opts.knob2.is_none();
    let shared_world =
        render_only_presentation.then(|| generate_tile_world(&opts, &opts.base_erosion));
    let shared_buffers = shared_world.as_ref().map(|world| {
        let mut buffers = if opts.display_subdivision_levels == 0 {
            generate_world_buffers(&gpu.device, &gpu.queue, world)
        } else {
            generate_world_buffers_with_display_subdivision(
                &gpu.device,
                &gpu.queue,
                world,
                opts.display_subdivision_levels,
            )
        };
        regenerate_surface_palette(&gpu.queue, world, &mut buffers, opts.surface_palette)
            .unwrap_or_else(|error| panic!("generic sweep palette: {error}"));
        buffers
    });

    for (ti, (overrides, label, fname, knob_values)) in tiles.iter().enumerate() {
        print!("[{}/{}] {label} ... ", ti + 1, n_tiles);
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let t0 = std::time::Instant::now();

        let owned_world =
            (!render_only_presentation).then(|| generate_tile_world(&opts, overrides));
        let world = shared_world
            .as_ref()
            .or(owned_world.as_ref())
            .expect("sweep world");

        if views.is_empty() {
            views = build_views(world, &opts);
            montage_w = opts.width * views.len() as u32;
            let montage_h = opts.height * n_tiles as u32;
            montage = vec![0u8; (montage_w * montage_h * 4) as usize];
        }

        let owned_buffers = (!render_only_presentation).then(|| {
            let mut buffers = if opts.display_subdivision_levels == 0 {
                generate_world_buffers(&gpu.device, &gpu.queue, world)
            } else {
                generate_world_buffers_with_display_subdivision(
                    &gpu.device,
                    &gpu.queue,
                    world,
                    opts.display_subdivision_levels,
                )
            };
            regenerate_surface_palette(&gpu.queue, world, &mut buffers, opts.surface_palette)
                .unwrap_or_else(|error| panic!("generic sweep palette: {error}"));
            buffers
        });
        let buffers = shared_buffers
            .as_ref()
            .or(owned_buffers.as_ref())
            .expect("sweep buffers");
        let mut image_filenames = Vec::with_capacity(views.len());
        for (vi, view) in views.iter().enumerate() {
            render_relief(
                &gpu,
                &mut renderer,
                &color_view,
                buffers,
                view.view_proj,
                view.eye,
                opts.river_mode,
                overrides.relief_scale.unwrap_or(RELIEF_SCALE),
                overrides.river_width_scale.unwrap_or(1.0),
            );
            let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
            let filename = format!("{fname}_{}.png", view.label);
            let tile_path = opts.out_dir.join(&filename);
            write_png(&tile_path, &rgba, opts.width, opts.height);
            image_filenames.push(filename);
            blit_tile(
                &mut montage,
                montage_w,
                &rgba,
                opts.width,
                opts.height,
                vi as u32,
                ti as u32,
            );
        }
        tile_records.push(TileRecord {
            index: ti,
            label: label.clone(),
            filename_stem: fname.clone(),
            knob_values: knob_values
                .iter()
                .map(|kv| KnobValue {
                    name: kv.name.clone(),
                    value: kv.value,
                })
                .collect(),
            relief_scale: overrides.relief_scale.unwrap_or(RELIEF_SCALE),
            river_width_scale: overrides.river_width_scale.unwrap_or(1.0),
            image_filenames,
            world_manifest: world.manifest(),
        });

        println!("{:.1}s", t0.elapsed().as_secs_f64());
    }

    let montage_h = opts.height * n_tiles as u32;
    let montage_path = opts.out_dir.join("montage.png");
    write_png(&montage_path, &montage, montage_w, montage_h);
    let sidecar = CaptureSidecar {
        schema_version: 1,
        coordinate_convention: "latitude=asin(y); longitude=atan2(z,x); degrees; positive longitude rotates +X toward +Z",
        config: CaptureConfig {
            seed: opts.seed,
            requested_coarse_cells: opts.cells,
            fine_scale: opts.fine_scale,
            fine_max: opts.fine_max,
            target_stage: opts.target_stage,
            voronoi_backend: opts.voronoi_backend,
            orogen_model: opts.orogen_model,
            fine_cache: opts.fine_cache,
            viewport_width: opts.width,
            viewport_height: opts.height,
            sweep_stack: opts.stack.clone(),
            primary_knob: opts.knob1.clone(),
            primary_values: opts.values1.clone(),
            secondary_knob: opts.knob2.clone(),
            secondary_values: opts.values2.clone(),
            overview_yaw_deg: opts.yaw_deg,
            overview_pitch_deg: opts.pitch_deg,
            overview_distance: opts.distance,
            closeup_altitude: opts.zoom_alt,
            explicit_targets: opts.targets.clone(),
            automatic_zoom_views_if_no_targets: opts.zoom_views,
            river_mode: river_mode_label(opts.river_mode),
            surface_palette: opts.surface_palette.name(),
            river_threshold_policy: opts.river_threshold_policy.clone(),
            river_min_catchment_km2: opts.river_min_catchment_km2,
            display_subdivision_levels: opts.display_subdivision_levels,
        },
        views: views.iter().map(|view| view.sidecar.clone()).collect(),
        tiles: tile_records,
        montage_filename: "montage.png",
        future_layers: ["Diagnostic-layer captures are not implemented in this packet."],
    };
    let sidecar_path = opts.out_dir.join("capture.json");
    let sidecar_file = std::fs::File::create(&sidecar_path)
        .unwrap_or_else(|e| panic!("create {}: {e}", sidecar_path.display()));
    serde_json::to_writer_pretty(BufWriter::new(sidecar_file), &sidecar)
        .expect("write capture sidecar");
    println!(
        "Done: {} tiles x {} views + montage + capture.json -> {}",
        n_tiles,
        views.len(),
        montage_path.display()
    );
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::{
        apply_knob, baseline_site_probe_config, build_stack_tiles, canonical_path_edges,
        compare_site_positions, diagnostic_coarse_support_site_probe_config,
        loose_site_probe_config, robust_scale, route_probe_config, select_readability_mouth,
        selected_orogen_model, tight_site_probe_config, ReadabilityMouthCandidate, SweepTarget,
    };
    use crate::app::coloring::{
        living_surface_blended_color, LIVING_HERBACEOUS_COLOR, LIVING_WETLAND_COLOR,
        LIVING_WOODY_COLOR,
    };
    use crate::app::world::ErosionOverrides;
    use hex3::world::{OrogenModel, PhysiognomyFractions};

    #[test]
    fn parses_dossier_target_in_project_coordinates() {
        let target: SweepTarget = "range_a:30:90".parse().unwrap();
        let p = target.unit_position();
        assert!((p.x - 0.0).abs() < 1.0e-6);
        assert!((p.y - 0.5).abs() < 1.0e-6);
        assert!((p.z - 30.0_f32.to_radians().cos()).abs() < 1.0e-6);
    }

    #[test]
    fn rejects_invalid_dossier_targets() {
        assert!("missing-fields".parse::<SweepTarget>().is_err());
        assert!("bad/id:0:0".parse::<SweepTarget>().is_err());
        assert!("north:91:0".parse::<SweepTarget>().is_err());
        assert!("wrap:0:181".parse::<SweepTarget>().is_err());
        assert!("nan:NaN:0".parse::<SweepTarget>().is_err());
    }

    #[test]
    fn robust_scale_ignores_non_finite_values_and_outliers() {
        let mut values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        values.extend([f32::NAN, f32::INFINITY, 100_000.0]);
        let (lo, hi) = robust_scale(&values);
        assert_eq!(lo, 2.0);
        assert_eq!(hi, 98.0);
    }

    #[test]
    fn robust_scale_expands_constant_fields() {
        let (lo, hi) = robust_scale(&[4.0; 8]);
        assert_eq!(lo, 4.0);
        assert!(hi > lo);
    }

    #[test]
    fn site_probe_prior_panel_is_valid_and_ordered() {
        let baseline = baseline_site_probe_config().validate().unwrap();
        let coarse = diagnostic_coarse_support_site_probe_config()
            .validate()
            .unwrap();
        let tight = tight_site_probe_config().validate().unwrap();
        let loose = loose_site_probe_config().validate().unwrap();
        assert!(tight.minimum_site_spacing_km > baseline.minimum_site_spacing_km);
        assert!(loose.minimum_site_spacing_km < baseline.minimum_site_spacing_km);
        assert!(
            tight.freshwater_access_limit_generalized_km
                < baseline.freshwater_access_limit_generalized_km
        );
        assert!(
            loose.freshwater_access_limit_generalized_km
                > baseline.freshwater_access_limit_generalized_km
        );
        assert_eq!(baseline.site_count, tight.site_count);
        assert_eq!(baseline.site_count, loose.site_count);
        assert_eq!(baseline.candidate_pool_size, 512);
        assert_eq!(coarse.candidate_pool_size, 160);
        assert_eq!(coarse.site_count, baseline.site_count);
        assert_eq!(
            coarse.maximum_total_catchment_cell_visits,
            baseline.maximum_total_catchment_cell_visits
        );
        route_probe_config().validate().unwrap();
    }

    #[test]
    fn route_path_edges_are_direction_independent_and_deduplicated() {
        let forward = canonical_path_edges(&[5, 2, 8, 2]);
        let reverse = canonical_path_edges(&[2, 8, 2, 5]);
        assert_eq!(forward, reverse);
        assert_eq!(forward.len(), 2);
        assert!(forward.contains(&(2, 5)));
        assert!(forward.contains(&(2, 8)));
    }

    #[test]
    fn site_comparison_reports_retention_and_directional_nearest_distances() {
        let baseline = [Vec3::X, Vec3::Y];
        let variant = [Vec3::X, Vec3::Z];
        let result = compare_site_positions(&baseline, &variant);
        let quarter_circumference = std::f32::consts::FRAC_PI_2 * hex3::world::PLANET_RADIUS_KM;
        assert_eq!(result.exact_retained_anchor_count, 1);
        assert_eq!(result.exact_retained_anchor_fraction, 0.5);
        assert_eq!(
            result.mean_nearest_baseline_site_distance_km,
            Some(quarter_circumference * 0.5)
        );
        assert_eq!(
            result.median_nearest_baseline_site_distance_km,
            Some(quarter_circumference * 0.5)
        );
        assert_eq!(
            result.maximum_nearest_baseline_site_distance_km,
            Some(quarter_circumference)
        );
    }

    #[test]
    fn living_surface_presentation_is_a_linear_fraction_blend() {
        let substrate = Vec3::new(0.7, 0.5, 0.3);
        for (fractions, expected) in [
            (
                PhysiognomyFractions {
                    bare: 1.0,
                    ..Default::default()
                },
                substrate,
            ),
            (
                PhysiognomyFractions {
                    herbaceous: 1.0,
                    ..Default::default()
                },
                LIVING_HERBACEOUS_COLOR,
            ),
            (
                PhysiognomyFractions {
                    woody: 1.0,
                    ..Default::default()
                },
                LIVING_WOODY_COLOR,
            ),
            (
                PhysiognomyFractions {
                    wetland: 1.0,
                    ..Default::default()
                },
                LIVING_WETLAND_COLOR,
            ),
        ] {
            assert_eq!(
                living_surface_blended_color(substrate, fractions, false),
                expected
            );
        }

        let submerged =
            living_surface_blended_color(substrate, PhysiognomyFractions::default(), true);
        assert_eq!(submerged, substrate);
    }

    #[test]
    fn swept_orogen_model_is_selected_before_stage_one() {
        let mut overrides = ErosionOverrides::default();
        apply_knob(&mut overrides, "orogen_model", 1.0).unwrap();
        assert_eq!(
            selected_orogen_model(OrogenModel::Legacy, &overrides),
            OrogenModel::LegacyYield
        );
    }

    #[test]
    fn o3a_stack_is_isolated_and_energy_matched_by_declared_controls() {
        let arms = build_stack_tiles("o3a", &ErosionOverrides::default());
        assert_eq!(arms.len(), 3);
        assert_eq!(arms[0].0.interior_relief, Some(0.0));
        assert_eq!(arms[0].0.meso_base_relief, Some(0.0));
        assert_eq!(arms[1].0.interior_relief, Some(0.0611));
        assert_eq!(arms[1].0.meso_base_relief, Some(0.0));
        assert_eq!(arms[2].0.interior_relief, Some(0.0));
        assert_eq!(arms[2].0.meso_base_relief, Some(0.05));
        for (arm, _, _) in arms {
            assert_eq!(arm.front_strike_weight, Some(0.0));
            assert_eq!(arm.margin_contrast, Some(0.0));
            assert_eq!(arm.emergent_lambda, Some(0.0));
            assert_eq!(arm.emergent_structured, Some(0.0));
            assert_eq!(arm.meso_relief, Some(0.0));
            assert_eq!(arm.meso_style, Some(1));
            assert_eq!(arm.meso_wavelength_km, Some(25.0));
        }
    }

    #[test]
    fn readability_mouth_prefers_major_then_discharge_then_lower_identity() {
        let candidate = |mouth_cell, discharge, major| ReadabilityMouthCandidate {
            mouth_cell,
            downstream_water_cell: mouth_cell + 100,
            water_body_index: 0,
            discharge,
            major,
        };
        let candidates = [
            candidate(3, 200.0, false),
            candidate(8, 100.0, true),
            candidate(5, 100.0, true),
        ];
        let (selected, fallback) = select_readability_mouth(&candidates).unwrap();
        assert_eq!(selected.mouth_cell, 5);
        assert_eq!(fallback, "none");
    }

    #[test]
    fn readability_mouth_discloses_non_major_fallback() {
        let candidates = [
            ReadabilityMouthCandidate {
                mouth_cell: 7,
                downstream_water_cell: 10,
                water_body_index: 1,
                discharge: 40.0,
                major: false,
            },
            ReadabilityMouthCandidate {
                mouth_cell: 2,
                downstream_water_cell: 11,
                water_body_index: 2,
                discharge: 60.0,
                major: false,
            },
        ];
        let (selected, fallback) = select_readability_mouth(&candidates).unwrap();
        assert_eq!(selected.mouth_cell, 2);
        assert_ne!(fallback, "none");
        assert!(select_readability_mouth(&[]).is_none());
    }
}
