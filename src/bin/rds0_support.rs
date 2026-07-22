//! Fixed source-only truth packet for Regional Deformation Support Slice RDS0.
//!
//! This binary deliberately has no scientific controls. It generates the one
//! selected seed/episode/source scale, closes source and mesh ledgers, and
//! writes a JSON packet plus a common-coordinate CPU diagnostic.

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use glam::Vec3;
use hex3::world::{
    build_regional_deformation_rds0_v0, collect_convergent_fronts, collect_plate_boundaries,
    conservative_signed_flux_front_rates_v0, evaluate_regional_deformation_frame_v0,
    evaluate_regional_deformation_static_control_v0, BoundaryEdgeId, ConvergentFrontSet,
    RegionalDeformationFrameLedgerV0, RegionalDeformationOmissionV0, RegionalDeformationProgramV0,
    RegionalDeformationRasterLedgerV0, RegionalDeformationRegimeV0, RunManifest, Tessellation,
    VoronoiBackend, World, COLLISION_WIDTH, NUM_PLATES_DEFAULT, PLANET_RADIUS_KM,
};
use kiddo::{KdTree, SquaredEuclidean};
use serde::Serialize;

const SEED: u64 = 8_675_309;
const COARSE_CELLS: usize = 100_000;
const LLOYD_ITERATIONS: usize = 1;
const EPISODE_ID: usize = 9;
const SIGNED_FLUX_SUBSTEPS: usize = 8;
const PANEL_WIDTH: u32 = 768;
const PANEL_HEIGHT: u32 = 384;
const PANEL_GAP: u32 = 4;
const PANEL_COUNT: u32 = 5;

#[derive(Debug, Parser)]
#[command(
    about = "Write the fixed RDS0 source-only support truth packet",
    disable_help_subcommand = true
)]
struct Cli {
    /// New directory receiving support.json and diagnostic.png.
    #[arg(long)]
    output_dir: PathBuf,
}

#[derive(Serialize)]
struct SupportReport<'a> {
    schema: &'static str,
    status: &'static str,
    elapsed_seconds: f64,
    manifest: RunManifest,
    manifest_note: &'static str,
    config: ExperimentConfig,
    selected_parent: SelectedParentReport,
    signed_flux_ledger: SignedFluxLedgerReport,
    program: &'a RegionalDeformationProgramV0,
    frame_ledgers: Vec<&'a RegionalDeformationFrameLedgerV0>,
    mesh_ledgers: Vec<NamedMeshLedger>,
    omissions: OmissionReport<'a>,
    hashes: HashReport,
    render: RenderMetadata,
}

#[derive(Clone, Copy, Serialize)]
struct ExperimentConfig {
    seed: u64,
    requested_coarse_cells: usize,
    lloyd_iterations: usize,
    plate_count: usize,
    episode_id: usize,
    signed_flux_sigma_km: f64,
    signed_flux_implicit_substeps: usize,
    generated_stages: &'static [&'static str],
    terrain_generated: bool,
    scientific_cli_controls: usize,
}

#[derive(Serialize)]
struct SelectedParentReport {
    id: [usize; 2],
    episode_id: usize,
    regime: &'static str,
    source_edge_count: usize,
    length_km: f32,
    corrected_positive_flux_km2_per_myr: f64,
}

#[derive(Serialize)]
struct SignedFluxLedgerReport {
    processed_segments: usize,
    processed_edges: usize,
    untouched_edges: usize,
    input_signed_km2_per_myr: f64,
    output_signed_km2_per_myr: f64,
    input_positive_km2_per_myr: f64,
    output_positive_km2_per_myr: f64,
    closure_residual_km2_per_myr: f64,
}

#[derive(Serialize)]
struct NamedMeshLedger {
    field: &'static str,
    ledger: JsonMeshLedger,
}

#[derive(Serialize)]
struct JsonMeshLedger {
    /// None identifies the static control, which has no relative RDS frame.
    frame_index: Option<usize>,
    requested_flux_km2_per_myr: f64,
    allocated_flux_km2_per_myr: f64,
    unallocated_flux_km2_per_myr: f64,
    closure_residual_km2_per_myr: f64,
    active_cell_count: usize,
    additive_overlap_cell_count: usize,
}

#[derive(Serialize)]
struct OmissionReport<'a> {
    program: &'a [RegionalDeformationOmissionV0],
    static_control: &'a [RegionalDeformationOmissionV0],
    frames: Vec<&'a [RegionalDeformationOmissionV0]>,
}

#[derive(Serialize)]
struct HashReport {
    coarse_sites_fnv1a64: String,
    selected_source_fnv1a64: String,
    program_fnv1a64: String,
    control_density_fnv1a64: String,
    frame_density_fnv1a64: Vec<String>,
    diagnostic_pixels_fnv1a64: String,
}

#[derive(Clone, Serialize)]
struct RenderMetadata {
    file: &'static str,
    width_px: u32,
    height_px: u32,
    panel_width_px: u32,
    panel_height_px: u32,
    panel_gap_px: u32,
    panel_order: [&'static str; 5],
    projection: &'static str,
    sampling: &'static str,
    coordinate_convention: &'static str,
    field: &'static str,
    shared_density_range_per_myr: [f64; 2],
    color_map: &'static str,
    physical_state_modified: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    run(cli)
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir(&cli.output_dir).map_err(|error| {
        format!(
            "output directory must not already exist ({}): {error}",
            cli.output_dir.display()
        )
    })?;
    let started = Instant::now();
    let mut world = World::new_with_options(
        SEED,
        COARSE_CELLS,
        LLOYD_ITERATIONS,
        VoronoiBackend::ConvexHull,
    );
    world.generate_plates(NUM_PLATES_DEFAULT);
    world.generate_crust();
    world.generate_dynamics();
    world.generate_features();

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
    )?;
    let sigma_km = f64::from(COLLISION_WIDTH) * f64::from(PLANET_RADIUS_KM);
    let corrected =
        conservative_signed_flux_front_rates_v0(&fronts, sigma_km, SIGNED_FLUX_SUBSTEPS)?;
    // The source module owns all RDS allocation and coarse-mesh evaluation.
    // This diagnostic only serializes and renders those authoritative results.
    let program = build_regional_deformation_rds0_v0(&fronts, EPISODE_ID)?;
    let control = evaluate_regional_deformation_static_control_v0(
        &program,
        &fronts,
        &world.tessellation,
        world.plates.as_ref().expect("plates generated"),
        world.crust.as_ref().expect("crust generated"),
    )?;
    let rasters = (0..program.frames.len())
        .map(|frame_index| {
            evaluate_regional_deformation_frame_v0(
                &program,
                frame_index,
                &fronts,
                &world.tessellation,
                world.plates.as_ref().expect("plates generated"),
                world.crust.as_ref().expect("crust generated"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let frame_fields: Vec<&[f64]> = rasters
        .iter()
        .map(|raster| raster.rate_density_per_myr.as_slice())
        .collect();
    if frame_fields.len() != 4 {
        return Err(format!(
            "RDS0 must emit four mesh frames, got {}",
            frame_fields.len()
        )
        .into());
    }
    let mut fields = Vec::with_capacity(5);
    fields.push(control.rate_density_per_myr.as_slice());
    fields.extend(frame_fields.iter().copied());
    let (pixels, render) = render_diagnostic(&world.tessellation, &fields)?;
    write_png(&cli.output_dir.join("diagnostic.png"), &pixels, &render)?;

    let selected_parent = SelectedParentReport {
        id: edge_pair(program.parent_segment_id),
        episode_id: program.episode_id,
        regime: regime_name(program.parent_regime),
        source_edge_count: program.parent_source_edges.len(),
        length_km: program.parent_length_km as f32,
        corrected_positive_flux_km2_per_myr: program.parent_positive_flux_km2_per_myr,
    };
    let ledger = corrected.ledger;
    let report = SupportReport {
        schema: "hex3.rds0-source-support.v0",
        status: "research-only reduced-kinematic counterfactual; not promoted terrain",
        elapsed_seconds: started.elapsed().as_secs_f64(),
        manifest: world.manifest(),
        manifest_note: "RunManifest computed_stage follows completed product stages; this intentionally partial source-only generation is enumerated by config.generated_stages",
        config: ExperimentConfig {
            seed: SEED,
            requested_coarse_cells: COARSE_CELLS,
            lloyd_iterations: LLOYD_ITERATIONS,
            plate_count: NUM_PLATES_DEFAULT,
            episode_id: EPISODE_ID,
            signed_flux_sigma_km: sigma_km,
            signed_flux_implicit_substeps: SIGNED_FLUX_SUBSTEPS,
            generated_stages: &["tessellation", "plates", "crust", "dynamics", "features"],
            terrain_generated: false,
            scientific_cli_controls: 0,
        },
        selected_parent,
        signed_flux_ledger: SignedFluxLedgerReport {
            processed_segments: ledger.processed_segment_count,
            processed_edges: ledger.processed_edge_count,
            untouched_edges: ledger.untouched_edge_count,
            input_signed_km2_per_myr: ledger.input_signed_flux_km2_per_myr,
            output_signed_km2_per_myr: ledger.output_signed_flux_km2_per_myr,
            input_positive_km2_per_myr: ledger.input_positive_clipped_flux_km2_per_myr,
            output_positive_km2_per_myr: ledger.output_positive_clipped_flux_km2_per_myr,
            closure_residual_km2_per_myr: ledger.closure_residual_km2_per_myr,
        },
        program: &program,
        frame_ledgers: program.frames.iter().map(|frame| &frame.ledger).collect(),
        mesh_ledgers: std::iter::once(NamedMeshLedger {
            field: "static-nearest-control",
            ledger: json_mesh_ledger(&control.ledger, None),
        })
        .chain(
            rasters
                .iter()
                .enumerate()
                .map(|(index, raster)| NamedMeshLedger {
                    field: [
                        "growth-0",
                        "growth-overlap-1",
                        "linkage-2",
                        "linked-successor-3",
                    ][index],
                    ledger: json_mesh_ledger(&raster.ledger, Some(index)),
                }),
        )
        .collect(),
        omissions: OmissionReport {
            program: &program.omissions,
            static_control: &control.omissions,
            frames: rasters
                .iter()
                .map(|raster| raster.omissions.as_slice())
                .collect(),
        },
        hashes: HashReport {
            coarse_sites_fnv1a64: hash_vec3s(&world.tessellation.voronoi.generators),
            selected_source_fnv1a64: hash_program_source(&program, &fronts),
            program_fnv1a64: hex_hash(fnv1a64(&serde_json::to_vec(&program)?)),
            control_density_fnv1a64: hash_f64s(&control.rate_density_per_myr),
            frame_density_fnv1a64: frame_fields.iter().map(|field| hash_f64s(field)).collect(),
            diagnostic_pixels_fnv1a64: hex_hash(fnv1a64(&pixels)),
        },
        render,
    };
    let json = serde_json::to_vec_pretty(&report)?;
    fs::write(cli.output_dir.join("support.json"), json)?;
    println!(
        "RDS0 source packet: {} (parent {:?}, {:.3} km²/Myr)",
        cli.output_dir.display(),
        program.parent_segment_id,
        report.selected_parent.corrected_positive_flux_km2_per_myr,
    );
    Ok(())
}

fn render_diagnostic(
    tessellation: &Tessellation,
    fields: &[&[f64]],
) -> Result<(Vec<u8>, RenderMetadata), Box<dyn std::error::Error>> {
    if fields.len() != PANEL_COUNT as usize
        || fields
            .iter()
            .any(|field| field.len() != tessellation.num_cells())
    {
        return Err("render requires five cell-aligned fields".into());
    }
    let shared_max = fields
        .iter()
        .flat_map(|field| field.iter().copied())
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max)
        .max(f64::EPSILON);
    let width = PANEL_COUNT * PANEL_WIDTH + (PANEL_COUNT - 1) * PANEL_GAP;
    let height = PANEL_HEIGHT;
    let mut pixels = vec![0_u8; width as usize * height as usize * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[9, 12, 18, 255]);
    }
    let mut tree = KdTree::<f32, 3>::with_capacity(tessellation.num_cells());
    for cell in 0..tessellation.num_cells() {
        tree.add(&tessellation.cell_center(cell).to_array(), cell as u64);
    }
    let lookup = render_lookup(PANEL_WIDTH, PANEL_HEIGHT, |point| {
        deterministic_nearest_cell(&tree, point)
    });
    for (panel, field) in fields.iter().enumerate() {
        let x0 = panel as u32 * (PANEL_WIDTH + PANEL_GAP);
        for y in 0..PANEL_HEIGHT {
            for x in 0..PANEL_WIDTH {
                let cell = lookup[(y * PANEL_WIDTH + x) as usize];
                let rgba = density_color(field[cell], shared_max);
                let offset = ((y * width + x0 + x) * 4) as usize;
                pixels[offset..offset + 4].copy_from_slice(&rgba);
            }
        }
    }
    Ok((
        pixels,
        RenderMetadata {
            file: "diagnostic.png",
            width_px: width,
            height_px: height,
            panel_width_px: PANEL_WIDTH,
            panel_height_px: PANEL_HEIGHT,
            panel_gap_px: PANEL_GAP,
            panel_order: ["static-nearest-control", "growth-0", "growth-overlap-1", "linkage-2", "linked-successor-3"],
            projection: "five matched uncropped 2:1 equirectangular whole-world panels",
            sampling: "deterministic nearest coarse Voronoi site at pixel centers; stable cell-index tie break",
            coordinate_convention: "latitude=asin(y); longitude=atan2(z,x); positive longitude rotates +X toward +Z",
            field: "opportunity rate density (1/Myr); one shared square-root display scale",
            shared_density_range_per_myr: [0.0, shared_max],
            color_map: "black-purple-cyan-yellow sequential; zero is near-black",
            physical_state_modified: false,
        },
    ))
}

fn render_lookup(width: u32, height: u32, mut nearest: impl FnMut(Vec3) -> usize) -> Vec<usize> {
    let mut lookup = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            lookup.push(nearest(equirectangular_pixel_point(x, y, width, height)));
        }
    }
    lookup
}

fn deterministic_nearest_cell(tree: &KdTree<f32, 3>, point: Vec3) -> usize {
    // At an exact Voronoi edge/vertex, Kiddo's traversal order need not define
    // identity. Eight candidates cover the maximum practical spherical
    // Voronoi tie here; exact equal distances use canonical cell index.
    let candidates = tree.nearest_n::<SquaredEuclidean>(&point.to_array(), 8);
    let minimum = candidates
        .first()
        .expect("coarse tessellation has at least one site")
        .distance;
    candidates
        .iter()
        .take_while(|candidate| candidate.distance == minimum)
        .map(|candidate| candidate.item as usize)
        .min()
        .expect("nearest query returned at least one site")
}

fn equirectangular_pixel_point(x: u32, y: u32, width: u32, height: u32) -> Vec3 {
    let longitude = ((f64::from(x) + 0.5) / f64::from(width) * std::f64::consts::TAU
        - std::f64::consts::PI) as f32;
    let latitude = (std::f64::consts::FRAC_PI_2
        - (f64::from(y) + 0.5) / f64::from(height) * std::f64::consts::PI)
        as f32;
    spherical_point(latitude, longitude)
}

fn spherical_point(latitude: f32, longitude: f32) -> Vec3 {
    let cos_latitude = latitude.cos();
    Vec3::new(
        cos_latitude * longitude.cos(),
        latitude.sin(),
        cos_latitude * longitude.sin(),
    )
}

fn density_color(value: f64, maximum: f64) -> [u8; 4] {
    let t = if value.is_finite() && value > 0.0 {
        (value / maximum).clamp(0.0, 1.0).sqrt()
    } else {
        0.0
    };
    let stops = [
        (0.0, [8.0, 10.0, 16.0]),
        (0.18, [51.0, 24.0, 92.0]),
        (0.48, [31.0, 153.0, 184.0]),
        (0.78, [95.0, 220.0, 168.0]),
        (1.0, [253.0, 231.0, 92.0]),
    ];
    let mut color = stops[stops.len() - 1].1;
    for pair in stops.windows(2) {
        if t <= pair[1].0 {
            let u = (t - pair[0].0) / (pair[1].0 - pair[0].0);
            color = [
                pair[0].1[0] + u * (pair[1].1[0] - pair[0].1[0]),
                pair[0].1[1] + u * (pair[1].1[1] - pair[0].1[1]),
                pair[0].1[2] + u * (pair[1].1[2] - pair[0].1[2]),
            ];
            break;
        }
    }
    [color[0] as u8, color[1] as u8, color[2] as u8, 255]
}

fn write_png(
    path: &Path,
    pixels: &[u8],
    metadata: &RenderMetadata,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut encoder =
        png::Encoder::new(BufWriter::new(file), metadata.width_px, metadata.height_px);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.add_text_chunk(
        "Title".into(),
        "Hex3 RDS0 source support truth packet".into(),
    )?;
    encoder.add_text_chunk("Panel order".into(), metadata.panel_order.join(", "))?;
    encoder.add_text_chunk("Field".into(), metadata.field.into())?;
    encoder.add_text_chunk("Projection".into(), metadata.projection.into())?;
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}

fn hash_program_source(
    program: &RegionalDeformationProgramV0,
    fronts: &ConvergentFrontSet,
) -> String {
    let by_id: std::collections::BTreeMap<_, _> =
        fronts.edges.iter().map(|edge| (edge.id, edge)).collect();
    let mut bytes = Vec::with_capacity(program.parent_source_edges.len() * 32);
    for edge_id in &program.parent_source_edges {
        let edge = by_id[edge_id];
        bytes.extend_from_slice(&(edge_id.cell_a as u64).to_le_bytes());
        bytes.extend_from_slice(&(edge_id.cell_b as u64).to_le_bytes());
        bytes.extend_from_slice(&edge.length_km.to_bits().to_le_bytes());
        let rate = program
            .corrected_signed_rates
            .iter()
            .find(|rate| rate.source_edge == *edge_id)
            .expect("program has a corrected rate for every source edge")
            .signed_rate_km_per_myr;
        bytes.extend_from_slice(&rate.to_bits().to_le_bytes());
    }
    hex_hash(fnv1a64(&bytes))
}

fn hash_vec3s(values: &[Vec3]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 12);
    for value in values {
        bytes.extend_from_slice(&value.x.to_bits().to_le_bytes());
        bytes.extend_from_slice(&value.y.to_bits().to_le_bytes());
        bytes.extend_from_slice(&value.z.to_bits().to_le_bytes());
    }
    hex_hash(fnv1a64(&bytes))
}

fn hash_f64s(values: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    hex_hash(fnv1a64(&bytes))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn hex_hash(hash: u64) -> String {
    format!("{hash:016x}")
}

fn json_mesh_ledger(
    ledger: &RegionalDeformationRasterLedgerV0,
    frame_index: Option<usize>,
) -> JsonMeshLedger {
    JsonMeshLedger {
        frame_index,
        requested_flux_km2_per_myr: ledger.requested_flux_km2_per_myr,
        allocated_flux_km2_per_myr: ledger.allocated_flux_km2_per_myr,
        unallocated_flux_km2_per_myr: ledger.unallocated_flux_km2_per_myr,
        closure_residual_km2_per_myr: ledger.closure_residual_km2_per_myr,
        active_cell_count: ledger.active_cell_count,
        additive_overlap_cell_count: ledger.additive_overlap_cell_count,
    }
}

fn edge_pair(id: BoundaryEdgeId) -> [usize; 2] {
    [id.cell_a, id.cell_b]
}

fn regime_name(regime: RegionalDeformationRegimeV0) -> &'static str {
    match regime {
        RegionalDeformationRegimeV0::Collision => "collision",
        RegionalDeformationRegimeV0::Subduction => "subduction",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn projection_places_cardinals_and_wraps_antimeridian() {
        assert!(spherical_point(0.0, 0.0).abs_diff_eq(Vec3::X, 1e-6));
        assert!(spherical_point(0.0, std::f32::consts::FRAC_PI_2).abs_diff_eq(Vec3::Z, 1e-6));
        assert!(spherical_point(std::f32::consts::FRAC_PI_2, 0.0).abs_diff_eq(Vec3::Y, 1e-6));
        assert!(spherical_point(0.0, std::f32::consts::PI)
            .abs_diff_eq(spherical_point(0.0, -std::f32::consts::PI), 1e-6));
        let width = 8;
        let height = 4;
        let west = equirectangular_pixel_point(0, 2, width, height);
        let east = equirectangular_pixel_point(width - 1, 2, width, height);
        assert!(
            west.dot(east) > 0.6,
            "antimeridian columns must be adjacent"
        );
        let upper = equirectangular_pixel_point(4, 0, width, height);
        let lower = equirectangular_pixel_point(4, height - 1, width, height);
        assert!(upper.y > 0.0 && lower.y < 0.0);
        let center_left = equirectangular_pixel_point(3, 1, width, height);
        let center_right = equirectangular_pixel_point(4, 1, width, height);
        assert!(center_left.x > 0.0 && center_right.x > 0.0);
    }

    #[test]
    fn nearest_lookup_is_deterministic_and_ties_use_lowest_index() {
        let sites = [Vec3::X, Vec3::Z];
        let nearest = |point: Vec3| {
            sites
                .iter()
                .enumerate()
                .min_by(|(ia, a), (ib, b)| {
                    point
                        .distance_squared(**a)
                        .total_cmp(&point.distance_squared(**b))
                        .then_with(|| ia.cmp(ib))
                })
                .unwrap()
                .0
        };
        let first = render_lookup(16, 8, nearest);
        let second = render_lookup(16, 8, nearest);
        assert_eq!(first, second);
        assert_eq!(nearest((Vec3::X + Vec3::Z).normalize()), 0);

        let mut tree = KdTree::<f32, 3>::new();
        tree.add(&Vec3::X.to_array(), 3);
        tree.add(&Vec3::Z.to_array(), 2);
        tree.add(&Vec3::NEG_X.to_array(), 1);
        assert_eq!(
            deterministic_nearest_cell(&tree, (Vec3::X + Vec3::Z).normalize()),
            2
        );
    }

    #[test]
    fn shared_scale_changes_equal_values_into_equal_pixels() {
        assert_eq!(density_color(0.25, 1.0), density_color(0.25, 1.0));
        assert_ne!(density_color(0.25, 1.0), density_color(0.5, 1.0));
        assert_eq!(density_color(f64::NAN, 1.0), density_color(0.0, 1.0));
    }

    #[test]
    fn cli_exposes_no_scientific_knobs() {
        let command = Cli::command();
        let long_names: Vec<_> = command
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .collect();
        assert_eq!(long_names, ["output-dir"]);
        assert!(Cli::try_parse_from(["rds0_support", "--seed", "1"]).is_err());
        assert!(Cli::try_parse_from(["rds0_support", "--output-dir", "out"]).is_ok());
    }
}
