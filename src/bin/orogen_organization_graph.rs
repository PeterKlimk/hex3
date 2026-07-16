//! Run the CPU-only B/F/I orogen organization compiler probe.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use hex3::world::landscape::orogen_organization_graph::{
    compile_organization_graph_v0, InheritanceModeV0, OrganizationCompilerConfigV0,
    OrganizationCompilerInputV0, OrganizationGraphProbeV0, OrganizationSourceLinkV0,
    OrganizationSourceSegmentV0, ParentWorkV0,
};
use hex3::world::landscape::{
    build_linked_shared_input_bundle_v0, LandscapeMesh, LinkedSharedInputBundleV0,
};
use serde::Serialize;

const FILE_SCHEMA_V0: &str = "orogen-organization-graph-probe-file-v0";

#[derive(Debug, Parser)]
#[command(about = "Compile and render the bounded 4 km B/F/I orogen forcing probe")]
struct Cli {
    /// New directory receiving organization.json and diagnostic.png.
    #[arg(long)]
    output_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct RenderMetadataV0 {
    image_width_px: u32,
    image_height_px: u32,
    map_panel_width_px: u32,
    map_height_px: u32,
    profile_row_height_px: u32,
    panel_order: [&'static str; 3],
    field: &'static str,
    shared_displacement_range_km: [f64; 2],
    profile_semantics: &'static str,
    physical_state_modified: bool,
    png_text_chunks: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct TimingsV0 {
    linked_input_seconds: f64,
    compiler_seconds: f64,
    render_seconds: f64,
    total_seconds: f64,
}

#[derive(Debug, Serialize)]
struct OutputFileV0<'a> {
    schema_version: &'static str,
    warning: &'static str,
    source_revision: &'static str,
    source_dirty: bool,
    semantic_hash_fnv1a64: String,
    timings: TimingsV0,
    render: RenderMetadataV0,
    probe: &'a OrganizationGraphProbeV0,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    fs::create_dir(&cli.output_dir)?;
    let total_started = Instant::now();

    let started = Instant::now();
    let bundle = build_linked_shared_input_bundle_v0()?;
    let linked_input_seconds = started.elapsed().as_secs_f64();
    let input = project_input(&bundle)?;

    let started = Instant::now();
    let probe = compile_organization_graph_v0(
        &input,
        OrganizationCompilerConfigV0::default(),
        InheritanceModeV0::CoherentLattice,
    )?;
    let compiler_seconds = started.elapsed().as_secs_f64();
    let semantic_hash_fnv1a64 = format!("{:016x}", fnv1a64(&bincode::serialize(&probe)?));

    let started = Instant::now();
    let render = write_diagnostic_png(
        &cli.output_dir.join("diagnostic.png"),
        &input.mesh,
        &input.source_segments,
        &probe,
    )?;
    let render_seconds = started.elapsed().as_secs_f64();
    let timings = TimingsV0 {
        linked_input_seconds,
        compiler_seconds,
        render_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    };
    let output = OutputFileV0 {
        schema_version: FILE_SCHEMA_V0,
        warning: "COMPILER-ONLY CAUSAL PROBE: displacement/work, not terrain or promotion evidence",
        source_revision: env!("HEX3_GIT_REVISION"),
        source_dirty: env!("HEX3_GIT_DIRTY") == "true",
        semantic_hash_fnv1a64,
        timings,
        render,
        probe: &probe,
    };
    let file = File::create(cli.output_dir.join("organization.json"))?;
    serde_json::to_writer_pretty(BufWriter::new(file), &output)?;
    Ok(())
}

fn project_input(
    bundle: &LinkedSharedInputBundleV0,
) -> Result<OrganizationCompilerInputV0, Box<dyn std::error::Error>> {
    let resolution = bundle
        .resolutions
        .iter()
        .find(|value| value.nominal_spacing_km.to_bits() == 4.0_f64.to_bits())
        .ok_or("accepted bundle has no exact 4 km input")?;
    let source_segments = bundle
        .declaration
        .scenario
        .segments
        .iter()
        .map(|segment| OrganizationSourceSegmentV0 {
            id: segment.id,
            start_km: [segment.geometry.start_km.x, segment.geometry.start_km.y],
            end_km: [segment.geometry.end_km.x, segment.geometry.end_km.y],
            width_km: segment.width_km,
            vergence_xy: [segment.vergence.x as f64, segment.vergence.y as f64],
            links: segment
                .links
                .iter()
                .map(|link| OrganizationSourceLinkV0 {
                    other: link.other,
                    kind: link.kind,
                })
                .collect(),
        })
        .collect();
    let parent_work_km3 = bundle
        .declaration
        .work_ledgers
        .iter()
        .filter_map(|ledger| {
            ledger.segment_id.map(|parent_id| ParentWorkV0 {
                parent_id,
                work_km3: ledger.positive_rock_volume_km3,
            })
        })
        .collect();
    Ok(OrganizationCompilerInputV0 {
        nominal_spacing_km: resolution.nominal_spacing_km,
        mesh: resolution.mesh.clone(),
        source_segments,
        baseline_stencils: resolution.compiled_stencils.clone(),
        parent_work_km3,
        total_work_km3: bundle.declaration.analytic_rock_volume_km3,
        source_bundle_hash: bundle.derived_bundle_hash,
        source_resolution_hash: resolution.derived_resolution_hash,
    })
}

fn write_diagnostic_png(
    path: &Path,
    mesh: &LandscapeMesh,
    source_segments: &[OrganizationSourceSegmentV0],
    probe: &OrganizationGraphProbeV0,
) -> Result<RenderMetadataV0, Box<dyn std::error::Error>> {
    const PANEL_W: u32 = 480;
    const MAP_H: u32 = 320;
    const PROFILE_H: u32 = 120;
    const GAP: u32 = 8;
    let width = 3 * PANEL_W + 2 * GAP;
    let height = MAP_H + 2 * PROFILE_H;
    let mut rgba = vec![0_u8; width as usize * height as usize * 4];
    for pixel in rgba.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[11, 14, 20, 255]);
    }
    let inherited = probe
        .inherited_displacement_km
        .as_ref()
        .ok_or("coherent fixture emitted no inherited displacement")?;
    let fields = [
        probe.baseline_displacement_km.as_slice(),
        probe.finite_displacement_km.as_slice(),
        inherited.as_slice(),
    ];
    let max = fields
        .iter()
        .flat_map(|field| field.iter().copied())
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);
    let bounds = mesh_bounds(mesh)?;
    for (panel, field) in fields.iter().enumerate() {
        paint_field(
            &mut rgba,
            width,
            panel as u32 * (PANEL_W + GAP),
            PANEL_W,
            MAP_H,
            bounds,
            mesh,
            field,
            max,
        );
    }
    // Make graph causality visible in I without altering the physical field.
    let inherited_x = 2 * (PANEL_W + GAP);
    for child in &probe.graph.children {
        let parent = source_segments
            .iter()
            .find(|segment| segment.id == child.parent_id)
            .ok_or("child references missing source parent")?;
        let point = point_along(parent, child.nucleus_km, probe.config.vergence_shift_km);
        let (x, y) = map_point(point, inherited_x, PANEL_W, MAP_H, bounds);
        paint_cross(&mut rgba, width, height, x, y, [255, 255, 255, 255]);
    }
    for (row, profile) in probe.profiles.iter().take(2).enumerate() {
        let y0 = MAP_H + row as u32 * PROFILE_H;
        paint_profile_row(
            &mut rgba,
            width,
            y0,
            PROFILE_H,
            profile,
            [125, 132, 145, 255],
            [53, 211, 238, 255],
            [251, 146, 60, 255],
        );
    }

    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.add_text_chunk(
        "Title".into(),
        "Orogen organization compiler: B/F/I displacement and work profiles".into(),
    )?;
    encoder.add_text_chunk(
        "Panel order".into(),
        "B accepted ribbon; F full-cosine parents; I inherited finite children".into(),
    )?;
    encoder.add_text_chunk(
        "White crosses".into(),
        "I-panel inherited child nuclei; presentation overlay only".into(),
    )?;
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&rgba)?;
    Ok(RenderMetadataV0 {
        image_width_px: width,
        image_height_px: height,
        map_panel_width_px: PANEL_W,
        map_height_px: MAP_H,
        profile_row_height_px: PROFILE_H,
        panel_order: ["B", "F", "I"],
        field: "cumulative physical rock displacement in km; one shared linear scale",
        shared_displacement_range_km: [0.0, max],
        profile_semantics: "area-integrated parent-attributed work by 8 km along-strike bin; B gray, F cyan, I orange",
        physical_state_modified: false,
        png_text_chunks: true,
    })
}

fn mesh_bounds(mesh: &LandscapeMesh) -> Result<[f64; 4], Box<dyn std::error::Error>> {
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for center in &mesh.cell_center_km {
        bounds[0] = bounds[0].min(center.x);
        bounds[1] = bounds[1].min(center.y);
        bounds[2] = bounds[2].max(center.x);
        bounds[3] = bounds[3].max(center.y);
    }
    if bounds.iter().all(|value| value.is_finite()) {
        Ok(bounds)
    } else {
        Err("mesh has no finite bounds".into())
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_field(
    rgba: &mut [u8],
    image_width: u32,
    x0: u32,
    panel_width: u32,
    panel_height: u32,
    bounds: [f64; 4],
    mesh: &LandscapeMesh,
    field: &[f64],
    shared_max: f64,
) {
    let mut raster = vec![f64::NAN; panel_width as usize * panel_height as usize];
    for (center, value) in mesh.cell_center_km.iter().zip(field) {
        let (px, py) = map_point([center.x, center.y], 0, panel_width, panel_height, bounds);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let x = px as i32 + dx;
                let y = py as i32 + dy;
                if x >= 0 && y >= 0 && x < panel_width as i32 && y < panel_height as i32 {
                    let slot = &mut raster[y as usize * panel_width as usize + x as usize];
                    *slot = if slot.is_nan() {
                        *value
                    } else {
                        slot.max(*value)
                    };
                }
            }
        }
    }
    for y in 0..panel_height {
        for x in 0..panel_width {
            let value = raster[y as usize * panel_width as usize + x as usize];
            if value.is_finite() {
                set_pixel(
                    rgba,
                    image_width,
                    x0 + x,
                    y,
                    displacement_color((value / shared_max).clamp(0.0, 1.0)),
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn paint_profile_row(
    rgba: &mut [u8],
    image_width: u32,
    y0: u32,
    row_height: u32,
    profile: &hex3::world::landscape::orogen_organization_graph::ParentLongitudinalProfileV0,
    b_color: [u8; 4],
    f_color: [u8; 4],
    i_color: [u8; 4],
) {
    let inherited = profile.inherited_work_km3.as_deref().unwrap_or(&[]);
    let max = profile
        .baseline_work_km3
        .iter()
        .chain(&profile.finite_work_km3)
        .chain(inherited)
        .copied()
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);
    for (values, color) in [
        (profile.baseline_work_km3.as_slice(), b_color),
        (profile.finite_work_km3.as_slice(), f_color),
        (inherited, i_color),
    ] {
        for index in 1..values.len() {
            let point = |i: usize| {
                let x = (i as f64 / (values.len() - 1) as f64 * (image_width - 1) as f64).round()
                    as i32;
                let y = y0 as i32 + row_height as i32
                    - 5
                    - (values[i] / max * (row_height - 10) as f64).round() as i32;
                (x, y)
            };
            draw_line(rgba, image_width, point(index - 1), point(index), color);
        }
    }
}

fn displacement_color(value: f64) -> [u8; 4] {
    let stops = [
        [18.0, 24.0, 38.0],
        [31.0, 111.0, 164.0],
        [49.0, 196.0, 141.0],
        [246.0, 210.0, 75.0],
        [239.0, 72.0, 54.0],
    ];
    let scaled = value * (stops.len() - 1) as f64;
    let index = (scaled.floor() as usize).min(stops.len() - 2);
    let t = scaled - index as f64;
    let mut out = [0_u8; 4];
    for channel in 0..3 {
        out[channel] =
            (stops[index][channel] * (1.0 - t) + stops[index + 1][channel] * t).round() as u8;
    }
    out[3] = 255;
    out
}

fn point_along(
    segment: &OrganizationSourceSegmentV0,
    arclength_km: f64,
    shift_km: f64,
) -> [f64; 2] {
    let start = glam::DVec2::from_array(segment.start_km);
    let axis = glam::DVec2::from_array(segment.end_km) - start;
    let unit = axis.normalize();
    let vergence = glam::DVec2::from_array(segment.vergence_xy).normalize_or_zero();
    (start + unit * arclength_km + vergence * shift_km).to_array()
}

fn map_point(point: [f64; 2], x0: u32, width: u32, height: u32, bounds: [f64; 4]) -> (u32, u32) {
    let x = ((point[0] - bounds[0]) / (bounds[2] - bounds[0]) * (width - 1) as f64)
        .clamp(0.0, (width - 1) as f64)
        .round() as u32;
    let y = ((bounds[3] - point[1]) / (bounds[3] - bounds[1]) * (height - 1) as f64)
        .clamp(0.0, (height - 1) as f64)
        .round() as u32;
    (x0 + x, y)
}

fn paint_cross(rgba: &mut [u8], width: u32, height: u32, x: u32, y: u32, color: [u8; 4]) {
    for offset in -5..=5 {
        for (dx, dy) in [(offset, offset), (offset, -offset)] {
            let px = x as i32 + dx;
            let py = y as i32 + dy;
            if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                set_pixel(rgba, width, px as u32, py as u32, color);
            }
        }
    }
}

fn draw_line(rgba: &mut [u8], width: u32, start: (i32, i32), end: (i32, i32), color: [u8; 4]) {
    let (mut x0, mut y0) = start;
    let (x1, y1) = end;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        if x0 >= 0 && y0 >= 0 {
            set_pixel(rgba, width, x0 as u32, y0 as u32, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let doubled = 2 * error;
        if doubled >= dy {
            error += dy;
            x0 += sx;
        }
        if doubled <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn set_pixel(rgba: &mut [u8], width: u32, x: u32, y: u32, color: [u8; 4]) {
    let offset = (y as usize * width as usize + x as usize) * 4;
    if offset + 4 <= rgba.len() {
        rgba[offset..offset + 4].copy_from_slice(&color);
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
