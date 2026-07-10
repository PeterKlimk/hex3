//! Compare two stage-separated world exports on one common coordinate set.
//!
//! The reference export supplies evaluation points and area weights. Candidate
//! fine fields are nearest-resampled onto those points, removing the biggest
//! confound in native-mesh A/B summaries. Generate stage-4 JSON exports with
//! identical seed/cell/fine-cap settings and different `--orogen-model` values.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use serde::Deserialize;

const EARTH_RADIUS_KM: f32 = 6371.0;
const METERS_PER_ELEVATION: f64 = 10_000.0;

#[derive(Parser)]
#[command(name = "compare-orogen-exports")]
struct Cli {
    /// Reference/baseline JSON export; its fine cells are the evaluation mesh.
    #[arg(long)]
    reference: PathBuf,
    /// Candidate JSON export resampled onto the reference cells.
    #[arg(long)]
    candidate: PathBuf,
}

#[derive(Deserialize)]
struct Export {
    metadata: Metadata,
    cells: Cells,
}

#[derive(Deserialize)]
struct Metadata {
    seed: u64,
    num_cells: usize,
    orogen_model: String,
}

#[derive(Deserialize)]
struct Cells {
    area: Vec<f32>,
    latitude: Vec<f32>,
    longitude: Vec<f32>,
    stages: Option<Stages>,
}

#[derive(Deserialize)]
struct Stages {
    coarse_envelope: Vec<f32>,
    fine_base: Vec<f32>,
    pre_erosion: Vec<f32>,
    eroded: Option<Vec<f32>>,
    structural_delta: Vec<f32>,
    erosion_delta: Option<Vec<f32>>,
    tectonic_thickening: Vec<f32>,
    tectonic_strain: Vec<f32>,
}

fn main() {
    let cli = Cli::parse();
    let reference = read_export(&cli.reference);
    let candidate = read_export(&cli.candidate);
    assert_eq!(
        reference.metadata.seed, candidate.metadata.seed,
        "exports must use the same seed"
    );
    let reference_stages = reference
        .cells
        .stages
        .as_ref()
        .expect("reference export has no fine stage arrays");
    let candidate_stages = candidate
        .cells
        .stages
        .as_ref()
        .expect("candidate export has no fine stage arrays");

    let candidate_points: Vec<[f32; 3]> = candidate
        .cells
        .latitude
        .iter()
        .zip(candidate.cells.longitude.iter())
        .map(|(&lat, &lon)| point(lat, lon))
        .collect();
    let tree = ImmutableKdTree::<f32, 3>::new_from_slice(&candidate_points);
    let mut mapping = Vec::with_capacity(reference.cells.latitude.len());
    let mut distances_km = Vec::with_capacity(mapping.capacity());
    for (&lat, &lon) in reference
        .cells
        .latitude
        .iter()
        .zip(reference.cells.longitude.iter())
    {
        let p = point(lat, lon);
        let nearest = tree.nearest_one::<SquaredEuclidean>(&p);
        mapping.push(nearest.item as usize);
        distances_km.push(nearest.distance.sqrt() * EARTH_RADIUS_KM);
    }
    distances_km.sort_by(f32::total_cmp);

    println!(
        "OROGEN COMMON-COORDINATE A/B seed={} reference={} ({} cells) candidate={} ({} cells)",
        reference.metadata.seed,
        reference.metadata.orogen_model,
        reference.metadata.num_cells,
        candidate.metadata.orogen_model,
        candidate.metadata.num_cells,
    );
    println!(
        "  resample nearest distance p50/p95/max = {:.2}/{:.2}/{:.2} km",
        percentile(&distances_km, 0.50),
        percentile(&distances_km, 0.95),
        distances_km.last().copied().unwrap_or(0.0),
    );

    compare_field(
        "coarse envelope",
        &reference_stages.coarse_envelope,
        &candidate_stages.coarse_envelope,
        &mapping,
        &reference.cells.area,
    );
    compare_field(
        "fine base",
        &reference_stages.fine_base,
        &candidate_stages.fine_base,
        &mapping,
        &reference.cells.area,
    );
    compare_field(
        "pre-erosion surface",
        &reference_stages.pre_erosion,
        &candidate_stages.pre_erosion,
        &mapping,
        &reference.cells.area,
    );
    compare_field(
        "structural delta",
        &reference_stages.structural_delta,
        &candidate_stages.structural_delta,
        &mapping,
        &reference.cells.area,
    );
    if let (Some(reference_eroded), Some(candidate_eroded)) =
        (&reference_stages.eroded, &candidate_stages.eroded)
    {
        compare_field(
            "eroded surface",
            reference_eroded,
            candidate_eroded,
            &mapping,
            &reference.cells.area,
        );
    }
    if let (Some(reference_erosion), Some(candidate_erosion)) = (
        &reference_stages.erosion_delta,
        &candidate_stages.erosion_delta,
    ) {
        compare_field(
            "erosion delta",
            reference_erosion,
            candidate_erosion,
            &mapping,
            &reference.cells.area,
        );
    }
    compare_field(
        "tectonic thickness",
        &reference_stages.tectonic_thickening,
        &candidate_stages.tectonic_thickening,
        &mapping,
        &reference.cells.area,
    );

    println!("  support retention on common coordinates:");
    for (label, reference_field, candidate_field) in [
        (
            "structural |delta|",
            &reference_stages.structural_delta,
            &candidate_stages.structural_delta,
        ),
        (
            "tectonic thickness+",
            &reference_stages.tectonic_thickening,
            &candidate_stages.tectonic_thickening,
        ),
    ] {
        for meters in [50.0f64, 100.0, 300.0, 600.0] {
            support_line(
                label,
                meters,
                reference_field,
                candidate_field,
                &mapping,
                &reference.cells.area,
                label.ends_with('+'),
            );
        }
    }

    // Strain has no legacy counterpart, but report its candidate coverage so
    // visual reviewers can distinguish "structure vanished" from "physical
    // strain exists but downstream gates ignored it".
    let candidate_strain: Vec<f32> = mapping
        .iter()
        .map(|&i| candidate_stages.tectonic_strain[i])
        .collect();
    let total_area: f64 = reference
        .cells
        .area
        .iter()
        .map(|&a| a.max(0.0) as f64)
        .sum();
    for threshold in [0.01f32, 0.05, 0.10] {
        let area: f64 = candidate_strain
            .iter()
            .zip(reference.cells.area.iter())
            .filter(|(strain, _)| **strain >= threshold)
            .map(|(_, &a)| a.max(0.0) as f64)
            .sum();
        let coverage = if area > 0.0 {
            100.0 * area / total_area.max(1e-20)
        } else {
            0.0
        };
        println!(
            "    candidate strain >= {:.2}: {:5.1}% planet area",
            threshold, coverage
        );
    }
}

fn read_export(path: &PathBuf) -> Export {
    serde_json::from_reader(BufReader::new(
        File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display())),
    ))
    .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn point(latitude: f32, longitude: f32) -> [f32; 3] {
    let cos_lat = latitude.cos();
    [
        cos_lat * longitude.cos(),
        latitude.sin(),
        cos_lat * longitude.sin(),
    ]
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[(((sorted.len() - 1) as f32) * p) as usize]
}

fn compare_field(
    label: &str,
    reference: &[f32],
    candidate: &[f32],
    mapping: &[usize],
    areas: &[f32],
) {
    let mut area = 0.0f64;
    let mut reference_energy = 0.0f64;
    let mut candidate_energy = 0.0f64;
    let mut difference_energy = 0.0f64;
    for i in 0..reference.len() {
        let a = areas[i].max(0.0) as f64;
        let r = reference[i] as f64 * METERS_PER_ELEVATION;
        let c = candidate[mapping[i]] as f64 * METERS_PER_ELEVATION;
        area += a;
        reference_energy += a * r * r;
        candidate_energy += a * c * c;
        difference_energy += a * (c - r) * (c - r);
    }
    println!(
        "  {label:<20} rms reference/candidate/difference = {:7.0}/{:7.0}/{:7.0} m, energy ratio={:.3}",
        (reference_energy / area).sqrt(),
        (candidate_energy / area).sqrt(),
        (difference_energy / area).sqrt(),
        candidate_energy / reference_energy.max(1e-20),
    );
}

#[allow(clippy::too_many_arguments)]
fn support_line(
    label: &str,
    meters: f64,
    reference: &[f32],
    candidate: &[f32],
    mapping: &[usize],
    areas: &[f32],
    positive_only: bool,
) {
    let threshold = meters / METERS_PER_ELEVATION;
    let active = |value: f32| {
        if positive_only {
            value as f64 >= threshold
        } else {
            (value as f64).abs() >= threshold
        }
    };
    let mut reference_area = 0.0f64;
    let mut retained_area = 0.0f64;
    let mut gained_area = 0.0f64;
    for i in 0..reference.len() {
        let a = areas[i].max(0.0) as f64;
        let r = active(reference[i]);
        let c = active(candidate[mapping[i]]);
        if r {
            reference_area += a;
            if c {
                retained_area += a;
            }
        } else if c {
            gained_area += a;
        }
    }
    println!(
        "    {label:<20} >= {:>3.0}m: retained {:5.1}% of reference support; gained {:5.1}% of reference-support area",
        meters,
        100.0 * retained_area / reference_area.max(1e-20),
        100.0 * gained_area / reference_area.max(1e-20),
    );
}
