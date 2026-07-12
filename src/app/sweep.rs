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

use glam::{Mat4, Vec3};
use serde::Serialize;

use hex3::render::{
    FillPipelineKind, GpuContext, IndexedDraw, OrbitCamera, RenderScene, Renderer, Uniforms,
};
use hex3::world::{FineCacheMode, OrogenModel, VoronoiBackend, World, RELIEF_SCALE};

use super::view::RiverMode;
use super::world::{
    advance_to_stage_2, advance_to_stage_3, advance_to_stage_3_with_cap, advance_to_stage_4,
    create_world_with_orogen_model, generate_world_buffers, ErosionOverrides,
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
    pub river_threshold_policy: String,
    pub river_min_catchment_km2: Option<f32>,
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

/// Generate a fully-staged world for one tile's knob values.
fn generate_tile_world(opts: &SweepOptions, overrides: &ErosionOverrides) -> World {
    let mut world = create_world_with_orogen_model(
        opts.seed,
        opts.cells,
        opts.voronoi_backend,
        opts.fine_cache,
        opts.orogen_model,
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
    river_threshold_policy: String,
    river_min_catchment_km2: Option<f32>,
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
        other => panic!("unknown --sweep-stack '{other}'; known: p1, v3, o0, meso"),
    }
}

/// Run the sweep: generate + render every knob combination to PNG tiles and a
/// stitched montage in `opts.out_dir`.
pub fn run_sweep(opts: SweepOptions) {
    if !opts.zoom_alt.is_finite() || opts.zoom_alt <= 0.0 {
        panic!("--sweep-zoom-alt must be finite and greater than zero");
    }
    let mut target_ids = std::collections::HashSet::new();
    for target in &opts.targets {
        if !target_ids.insert(target.id.as_str()) {
            panic!("duplicate --sweep-target id '{}'", target.id);
        }
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
    let shared_buffers = shared_world
        .as_ref()
        .map(|world| generate_world_buffers(&gpu.device, &gpu.queue, world));

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
            views = build_views(&world, &opts);
            montage_w = opts.width * views.len() as u32;
            let montage_h = opts.height * n_tiles as u32;
            montage = vec![0u8; (montage_w * montage_h * 4) as usize];
        }

        let owned_buffers = (!render_only_presentation)
            .then(|| generate_world_buffers(&gpu.device, &gpu.queue, world));
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
                &buffers,
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
            river_threshold_policy: opts.river_threshold_policy.clone(),
            river_min_catchment_km2: opts.river_min_catchment_km2,
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
    use super::SweepTarget;

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
}
