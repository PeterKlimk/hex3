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

use glam::{Mat4, Vec3};

use hex3::render::{
    FillPipelineKind, GpuContext, IndexedDraw, OrbitCamera, RenderScene, Renderer, Uniforms,
};
use hex3::world::{FineCacheMode, VoronoiBackend, World};

use super::view::RiverMode;
use super::world::{
    advance_to_stage_2, advance_to_stage_3, advance_to_stage_4, create_world_with_options,
    generate_world_buffers, ErosionOverrides,
};

/// Knobs the sweep can vary, mapped onto [`ErosionOverrides`] fields.
pub const SWEEP_KNOBS: &[&str] = &[
    "k",
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
    "lake_evap",
    "glacial_k",
    "fault_scarp",
    "interior_relief",
    "front_strike_weight",
    "margin_contrast",
    "emergent_lambda",
];

/// Options for a sweep run, assembled from the CLI.
pub struct SweepOptions {
    pub seed: u64,
    pub cells: usize,
    pub fine_scale: f32,
    pub target_stage: u32,
    pub voronoi_backend: VoronoiBackend,
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
    pub river_mode: RiverMode,
}

/// Apply one knob=value onto the overrides. Errors on an unknown knob name.
fn apply_knob(ov: &mut ErosionOverrides, name: &str, v: f64) -> Result<(), String> {
    let f = v as f32;
    match name {
        "k" => ov.k = Some(f),
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
        "lake_evap" => ov.lake_evap_strength = Some(f),
        "glacial_k" => ov.glacial_k = Some(f),
        "fault_scarp" => ov.fault_scarp_height = Some(f),
        "interior_relief" => ov.interior_relief = Some(f),
        "front_strike_weight" => ov.front_strike_weight = Some(f),
        "margin_contrast" => ov.margin_contrast = Some(f),
        "emergent_lambda" => ov.emergent_lambda = Some(f),
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
    let mut world =
        create_world_with_options(opts.seed, opts.cells, opts.voronoi_backend, opts.fine_cache);
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
        advance_to_stage_3(&mut world);
    }
    if opts.target_stage >= 4 {
        advance_to_stage_4(&mut world);
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
) {
    let light = Vec3::new(0.5, 1.0, 0.3).normalize();
    let uniforms = Uniforms::new(view_proj, cam_pos, light)
        .with_relief(true)
        .with_hemisphere_lighting(true)
        .with_map_mode(false);

    let river_mesh = match river_mode {
        RiverMode::Off => None,
        RiverMode::Major => (buffers.num_river_mesh_major_indices > 0).then_some(IndexedDraw {
            vertex_buffer: &buffers.river_mesh_major_vertex_buffer,
            index_buffer: &buffers.river_mesh_major_index_buffer,
            index_count: buffers.num_river_mesh_major_indices,
        }),
        RiverMode::All => (buffers.num_river_mesh_all_indices > 0).then_some(IndexedDraw {
            vertex_buffer: &buffers.river_mesh_all_vertex_buffer,
            index_buffer: &buffers.river_mesh_all_index_buffer,
            index_count: buffers.num_river_mesh_all_indices,
        }),
    };

    let scene = RenderScene {
        fill_pipeline: FillPipelineKind::UnifiedGlobe,
        fill: IndexedDraw {
            vertex_buffer: &buffers.unified_vertex_buffer,
            index_buffer: &buffers.unified_index_buffer,
            index_count: buffers.num_unified_indices,
        },
        edges: None,
        arrows: None,
        pole_markers: None,
        rivers: None,
        river_mesh,
        wind_particles: None,
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

/// Build the per-tile view set: a globe overview plus `zoom_views` close-ups
/// aimed at the highest land in `world`. The same set is reused for every tile so
/// each montage column shows the same region/angle across knob values.
fn build_views(world: &World, opts: &SweepOptions) -> Vec<(Mat4, Vec3, String)> {
    let aspect = opts.width as f32 / opts.height as f32;
    let mut views = Vec::new();

    let mut cam = OrbitCamera::new();
    cam.yaw = opts.yaw_deg.to_radians();
    cam.pitch = opts.pitch_deg.to_radians();
    cam.distance = opts.distance;
    cam.aspect = aspect;
    views.push((
        cam.view_projection(),
        cam.eye_position(),
        "globe".to_string(),
    ));

    for (i, t) in pick_targets(world, opts.zoom_views).iter().enumerate() {
        let (vp, eye) = target_camera(*t, aspect, opts.zoom_alt);
        views.push((vp, eye, format!("zoom{}", i + 1)));
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
    let emergent = |lambda: f32, uplift: f32, steps: usize, label: &str, fname: &str| {
        let mut o = *base;
        o.emergent_lambda = Some(lambda);
        o.interior_relief = Some(0.005); // faint seed
        o.front_strike_weight = Some(0.0);
        o.margin_contrast = Some(0.0);
        o.uplift_scale = Some(uplift);
        o.steps = Some(steps);
        o.mfd_exponent = Some(1.0);
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
        // erosion-v3: painted P1a (current best) vs emergent build at λ=0.25/0.5/0.75,
        // uplift_scale = λ/(steps·dt) with steps=120 so each rebuilds to ~target height.
        "v3" => vec![
            tile(0.04, 0.7, 1.0, "P1 painted (current)", "0_painted"),
            emergent(0.25, 0.25 / 120.0, 120, "emergent λ=0.25", "1_emergent_025"),
            emergent(0.5, 0.5 / 120.0, 120, "emergent λ=0.5", "2_emergent_050"),
            emergent(0.75, 0.75 / 120.0, 120, "emergent λ=0.75", "3_emergent_075"),
        ],
        other => panic!("unknown --sweep-stack '{other}'; known: p1, v3"),
    }
}

/// Run the sweep: generate + render every knob combination to PNG tiles and a
/// stitched montage in `opts.out_dir`.
pub fn run_sweep(opts: SweepOptions) {
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
    let tiles: Vec<(ErosionOverrides, String, String)> = if let Some(stack) = &opts.stack {
        build_stack_tiles(stack, &opts.base_erosion)
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
                if let (Some(k2), Some(v2)) = (&opts.knob2, row_val) {
                    apply_knob(&mut overrides, k2, *v2).unwrap();
                    label = format!("{label}, {}={}", k2, fmt_value(*v2));
                    fname = format!("{fname}__{}_{}", k2, fmt_value(*v2));
                }
                tiles.push((overrides, label, fname));
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
        1 + opts.zoom_views,
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
    let mut views: Vec<(Mat4, Vec3, String)> = Vec::new();
    let mut montage: Vec<u8> = Vec::new();
    let mut montage_w = 0u32;

    for (ti, (overrides, label, fname)) in tiles.iter().enumerate() {
        print!("[{}/{}] {label} ... ", ti + 1, n_tiles);
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let t0 = std::time::Instant::now();

        let world = generate_tile_world(&opts, overrides);

        if views.is_empty() {
            views = build_views(&world, &opts);
            montage_w = opts.width * views.len() as u32;
            let montage_h = opts.height * n_tiles as u32;
            montage = vec![0u8; (montage_w * montage_h * 4) as usize];
        }

        let buffers = generate_world_buffers(&gpu.device, &world);
        for (vi, (view_proj, eye, vlabel)) in views.iter().enumerate() {
            render_relief(
                &gpu,
                &mut renderer,
                &color_view,
                &buffers,
                *view_proj,
                *eye,
                opts.river_mode,
            );
            let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);
            let tile_path = opts.out_dir.join(format!("{fname}_{vlabel}.png"));
            write_png(&tile_path, &rgba, opts.width, opts.height);
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

        println!("{:.1}s", t0.elapsed().as_secs_f64());
    }

    let montage_h = opts.height * n_tiles as u32;
    let montage_path = opts.out_dir.join("montage.png");
    write_png(&montage_path, &montage, montage_w, montage_h);
    println!(
        "Done: {} tiles x {} views + montage -> {}",
        n_tiles,
        views.len(),
        montage_path.display()
    );
}
