//! Headless parameter-sweep harness: render the stage-4 relief view to PNG tiles
//! for a grid of erosion-knob values, plus a stitched contour-sheet montage, so
//! knob effects can be eyeballed without driving the interactive app frame by
//! frame. Runs on the GPU but needs no window (offscreen render-to-texture +
//! readback), so a single `cargo run` produces the whole image grid.
//!
//! The fine-mesh base (stage 3a) is a function of seed + mesh density only, so it
//! is disk-cached and reused across every tile; only the erosion stage re-runs
//! per knob value, which is what makes a sweep affordable after the perf work.

use std::io::BufWriter;
use std::path::PathBuf;

use glam::Vec3;

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
    /// Knob varied across columns and its values.
    pub knob1: String,
    pub values1: Vec<f64>,
    /// Optional knob varied across rows (grid); empty values => 1-D sweep.
    pub knob2: Option<String>,
    pub values2: Vec<f64>,
    pub out_dir: PathBuf,
    pub width: u32,
    pub height: u32,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub distance: f32,
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

/// Render a world's relief view into `color_view` (the offscreen target).
fn render_relief(
    gpu: &GpuContext,
    renderer: &mut Renderer,
    color_view: &wgpu::TextureView,
    world: &World,
    cam: &OrbitCamera,
    river_mode: RiverMode,
) {
    let buffers = generate_world_buffers(&gpu.device, world);

    let light = Vec3::new(0.5, 1.0, 0.3).normalize();
    let uniforms = Uniforms::new(cam.view_projection(), cam.eye_position(), light)
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

/// Run the sweep: generate + render every knob combination to PNG tiles and a
/// stitched montage in `opts.out_dir`.
pub fn run_sweep(opts: SweepOptions) {
    // Validate knob names up front so a typo fails before any (slow) generation.
    {
        let mut probe = ErosionOverrides::default();
        apply_knob(&mut probe, &opts.knob1, opts.values1[0]).unwrap_or_else(|e| panic!("{e}"));
        if let Some(k2) = &opts.knob2 {
            apply_knob(&mut probe, k2, opts.values2[0]).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    std::fs::create_dir_all(&opts.out_dir)
        .unwrap_or_else(|e| panic!("create {}: {e}", opts.out_dir.display()));

    // Rows: the second knob's values, or a single row for a 1-D sweep.
    let rows: Vec<Option<f64>> = if opts.knob2.is_some() && !opts.values2.is_empty() {
        opts.values2.iter().map(|&v| Some(v)).collect()
    } else {
        vec![None]
    };
    let cols = opts.values1.len();
    let n_tiles = cols * rows.len();

    println!(
        "Sweep: {} ({} values){} -> {} tiles at {}x{}, stage {}, seed {}",
        opts.knob1,
        cols,
        opts.knob2
            .as_ref()
            .map(|k| format!(" x {} ({} values)", k, rows.len()))
            .unwrap_or_default(),
        n_tiles,
        opts.width,
        opts.height,
        opts.target_stage,
        opts.seed,
    );

    // Headless GPU + renderer + offscreen color target (depth lives in Renderer).
    let gpu = pollster::block_on(GpuContext::new_headless(opts.width, opts.height));
    let init_uniforms = Uniforms::new(glam::Mat4::IDENTITY, Vec3::ZERO, Vec3::Y);
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

    let mut cam = OrbitCamera::new();
    cam.yaw = opts.yaw_deg.to_radians();
    cam.pitch = opts.pitch_deg.to_radians();
    cam.distance = opts.distance;
    cam.aspect = opts.width as f32 / opts.height as f32;

    let montage_w = opts.width * cols as u32;
    let montage_h = opts.height * rows.len() as u32;
    let mut montage = vec![0u8; (montage_w * montage_h * 4) as usize];

    let mut done = 0usize;
    for (ri, row_val) in rows.iter().enumerate() {
        for (ci, &v1) in opts.values1.iter().enumerate() {
            let mut overrides = opts.base_erosion;
            apply_knob(&mut overrides, &opts.knob1, v1).unwrap();
            let mut label = format!("{}={}", opts.knob1, fmt_value(v1));
            let mut fname = format!("{}_{}", opts.knob1, fmt_value(v1));
            if let (Some(k2), Some(v2)) = (&opts.knob2, row_val) {
                apply_knob(&mut overrides, k2, *v2).unwrap();
                label = format!("{label}, {}={}", k2, fmt_value(*v2));
                fname = format!("{fname}__{}_{}", k2, fmt_value(*v2));
            }

            done += 1;
            print!("[{done}/{n_tiles}] {label} ... ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
            let t0 = std::time::Instant::now();

            let world = generate_tile_world(&opts, &overrides);
            render_relief(
                &gpu,
                &mut renderer,
                &color_view,
                &world,
                &cam,
                opts.river_mode,
            );
            let rgba = read_back_rgba(&gpu, &color_tex, opts.width, opts.height);

            let tile_path = opts.out_dir.join(format!("{fname}.png"));
            write_png(&tile_path, &rgba, opts.width, opts.height);
            blit_tile(
                &mut montage,
                montage_w,
                &rgba,
                opts.width,
                opts.height,
                ci as u32,
                ri as u32,
            );

            println!(
                "{:.1}s -> {}",
                t0.elapsed().as_secs_f64(),
                tile_path.display()
            );
        }
    }

    let montage_path = opts.out_dir.join("montage.png");
    write_png(&montage_path, &montage, montage_w, montage_h);
    println!(
        "Done: {} tiles + montage -> {}",
        n_tiles,
        montage_path.display()
    );
}
