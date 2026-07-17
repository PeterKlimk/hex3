//! Bind the fixed source-first mountain target to unchanged legacy observations.

use std::collections::VecDeque;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use hex3::world::{
    bind_legacy_observations_to_source, catalog_structural_source_belts, collect_convergent_fronts,
    collect_plate_boundaries, compile_structural_mountain, elevation_to_km,
    select_primary_structural_source_belt, solid_angle_to_km2, BoundaryEdgeId, FineCacheMode,
    RunManifest, Tessellation, VoronoiBackend, World, NUM_PLATES_DEFAULT,
    STRUCTURAL_TARGET_THRESHOLD_KM,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Bind a source-first mountain target to legacy terrain observations")]
struct Cli {
    #[arg(long, default_value_t = 12_345)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    cells: usize,
    #[arg(long, default_value_t = 250_000)]
    fine_max: usize,
    #[arg(long)]
    rebuild_cache: bool,
    #[arg(
        long,
        default_value = "docs/generated/structural-mountain-seed-12345-observation-binding-v0.json"
    )]
    output: PathBuf,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    seed: u64,
    requested_coarse_cells: usize,
    actual_fine_cells: usize,
    elapsed_seconds: f32,
    manifest: RunManifest,
    source: SourceReport,
    binding: BindingReport,
    legacy_observation: ObservationReport,
}

#[derive(Serialize)]
struct SourceReport {
    id: [usize; 2],
    episode_id: usize,
    plate_pair: [usize; 2],
    source_edge_count: usize,
    source_edge_hash: String,
    source_edges: Vec<[usize; 2]>,
    segment_count: usize,
    length_km: f32,
    declared_opportunity_km2: f64,
}

#[derive(Serialize)]
struct BindingReport {
    legacy_eligible_source_edges: Vec<[usize; 2]>,
    unrepresented_source_edges: Vec<[usize; 2]>,
    mixed_seed_external_edges: Vec<[usize; 2]>,
    coarse_strict_response_cell_count: usize,
    coarse_mixed_response_cell_count: usize,
    fine_strict_owned_cell_count: usize,
    fine_strict_owned_area_km2: f64,
    fine_strict_owned_hash: String,
    fine_ambiguous_association_cell_count: usize,
    fine_ambiguous_association_area_km2: f64,
    fine_ambiguous_association_hash: String,
    max_response_reconstruction_error: f32,
    area_weighted_response_reconstruction_residual: f64,
    area_weighted_response_integral: ResponseIntegralReport,
    fine_strict_owned_cells: Vec<usize>,
    fine_ambiguous_association_cells: Vec<usize>,
}

#[derive(Serialize)]
struct ResponseIntegralReport {
    strict: f64,
    mixed: f64,
    other: f64,
    strict_fraction: f64,
    mixed_fraction: f64,
}

#[derive(Serialize)]
struct ObservationReport {
    visible_threshold_km: f32,
    strict_domain_peak_cell: usize,
    strict_domain_peak_elevation_km: f32,
    strict_domain_peak_latitude_deg: f32,
    strict_domain_peak_longitude_deg: f32,
    strict_domain_land_area_km2: f64,
    strict_domain_visible_cell_count: usize,
    strict_domain_visible_area_km2: f64,
    visible_component_count: usize,
    largest_visible_component_cell_count: usize,
    largest_visible_component_area_km2: f64,
    largest_visible_component_hash: String,
    largest_visible_component_cells: Vec<usize>,
    strict_domain_integration_cut_cells: usize,
    strict_domain_breached_source_cells: usize,
    visible_integration_cut_cells: usize,
    visible_breached_source_cells: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    let started = Instant::now();
    let mut world = World::new_with_options(cli.seed, cli.cells, 1, VoronoiBackend::ConvexHull);
    world.fine_cache = if cli.rebuild_cache {
        FineCacheMode::Rebuild
    } else {
        FineCacheMode::Enabled
    };
    world.generate_all(NUM_PLATES_DEFAULT);
    world.generate_atmosphere();
    world.generate_hydrology_with_fine_cap(cli.fine_max);

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
    let graph = compile_structural_mountain(&fronts)?;
    let belts = catalog_structural_source_belts(&graph)?;
    let source = select_primary_structural_source_belt(&belts)?;

    let fine = world.fine.as_ref().expect("fine world generated");
    let surface = fine.eroded.as_ref().expect("stage 4 generated");
    let binding = bind_legacy_observations_to_source(
        &world.tessellation,
        &fine.base.tessellation,
        &fine.base.coarse_cell,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.features.as_ref().expect("features generated"),
        &boundaries,
        &fronts,
        &source.source_edges,
    )?;

    let areas = fine.base.tessellation.cell_areas();
    let area = |cells: &[usize]| {
        cells
            .iter()
            .map(|&cell| f64::from(solid_angle_to_km2(areas[cell])))
            .sum::<f64>()
    };
    let response_integral = |values: &[f32]| {
        values
            .iter()
            .zip(&areas)
            .map(|(&value, &solid_angle)| {
                f64::from(value) * f64::from(solid_angle_to_km2(solid_angle))
            })
            .sum::<f64>()
    };
    let strict_integral = response_integral(&binding.strict_response);
    let mixed_integral = response_integral(&binding.mixed_response);
    let other_integral = response_integral(&binding.other_response);
    let response_total = strict_integral + mixed_integral + other_integral;
    let mut max_response_reconstruction_error = 0.0f32;
    let mut response_reconstruction_residual = 0.0f64;
    for cell in 0..fine.base.tessellation.num_cells() {
        let reconstructed = binding.strict_response[cell]
            + binding.mixed_response[cell]
            + binding.other_response[cell];
        let product = fine.base.fields.elevation_fields.arc[cell]
            + fine.base.fields.elevation_fields.collision[cell];
        let error = reconstructed - product;
        max_response_reconstruction_error = max_response_reconstruction_error.max(error.abs());
        response_reconstruction_residual +=
            f64::from(error) * f64::from(solid_angle_to_km2(areas[cell]));
    }

    let mut strict_mask = vec![false; fine.base.tessellation.num_cells()];
    for &cell in &binding.fine_strict_owned_cells {
        strict_mask[cell] = true;
    }
    let visible_cells: Vec<_> = binding
        .fine_strict_owned_cells
        .iter()
        .copied()
        .filter(|&cell| {
            elevation_to_km(surface.elevation.values[cell]) >= STRUCTURAL_TARGET_THRESHOLD_KM
        })
        .collect();
    let mut visible_mask = vec![false; strict_mask.len()];
    for &cell in &visible_cells {
        visible_mask[cell] = true;
    }
    let mut visible_components = connected_components(&fine.base.tessellation, &visible_mask);
    visible_components.sort_by(|left, right| {
        area(right)
            .total_cmp(&area(left))
            .then_with(|| left[0].cmp(&right[0]))
    });
    let largest_visible = visible_components.first().cloned().unwrap_or_default();
    let peak_cell = *binding
        .fine_strict_owned_cells
        .iter()
        .max_by(|&&left, &&right| {
            surface.elevation.values[left]
                .total_cmp(&surface.elevation.values[right])
                .then_with(|| right.cmp(&left))
        })
        .ok_or("source binding produced no strict fine cells")?;
    let peak_position = fine.base.tessellation.cell_center(peak_cell);
    let land_cells: Vec<_> = binding
        .fine_strict_owned_cells
        .iter()
        .copied()
        .filter(|&cell| !surface.hydrology.is_submerged(cell))
        .collect();
    let count = |cells: &[usize], predicate: &dyn Fn(usize) -> bool| {
        cells.iter().filter(|&&cell| predicate(cell)).count()
    };

    let report = Report {
        schema: "hex3-structural-mountain-observation-binding-v0",
        seed: cli.seed,
        requested_coarse_cells: cli.cells,
        actual_fine_cells: fine.base.tessellation.num_cells(),
        elapsed_seconds: started.elapsed().as_secs_f32(),
        manifest: RunManifest::from_world(&world),
        source: SourceReport {
            id: edge_pair(source.id),
            episode_id: source.episode_id,
            plate_pair: source.plate_pair,
            source_edge_count: source.source_edges.len(),
            source_edge_hash: hash_edges(&source.source_edges),
            source_edges: source.source_edges.iter().copied().map(edge_pair).collect(),
            segment_count: source.segment_ids.len(),
            length_km: source.length_km,
            declared_opportunity_km2: source.declared_opportunity_km2,
        },
        binding: BindingReport {
            legacy_eligible_source_edges: binding
                .legacy_eligible_source_edges
                .iter()
                .copied()
                .map(edge_pair)
                .collect(),
            unrepresented_source_edges: binding
                .unrepresented_source_edges
                .iter()
                .copied()
                .map(edge_pair)
                .collect(),
            mixed_seed_external_edges: binding
                .mixed_seed_external_edges
                .iter()
                .copied()
                .map(edge_pair)
                .collect(),
            coarse_strict_response_cell_count: binding.coarse_strict_response_cells.len(),
            coarse_mixed_response_cell_count: binding.coarse_mixed_response_cells.len(),
            fine_strict_owned_cell_count: binding.fine_strict_owned_cells.len(),
            fine_strict_owned_area_km2: area(&binding.fine_strict_owned_cells),
            fine_strict_owned_hash: hash_cells(&binding.fine_strict_owned_cells),
            fine_ambiguous_association_cell_count: binding.fine_ambiguous_association_cells.len(),
            fine_ambiguous_association_area_km2: area(&binding.fine_ambiguous_association_cells),
            fine_ambiguous_association_hash: hash_cells(&binding.fine_ambiguous_association_cells),
            max_response_reconstruction_error,
            area_weighted_response_reconstruction_residual: response_reconstruction_residual,
            area_weighted_response_integral: ResponseIntegralReport {
                strict: strict_integral,
                mixed: mixed_integral,
                other: other_integral,
                strict_fraction: strict_integral / response_total,
                mixed_fraction: mixed_integral / response_total,
            },
            fine_strict_owned_cells: binding.fine_strict_owned_cells.clone(),
            fine_ambiguous_association_cells: binding.fine_ambiguous_association_cells.clone(),
        },
        legacy_observation: ObservationReport {
            visible_threshold_km: STRUCTURAL_TARGET_THRESHOLD_KM,
            strict_domain_peak_cell: peak_cell,
            strict_domain_peak_elevation_km: elevation_to_km(surface.elevation.values[peak_cell]),
            strict_domain_peak_latitude_deg: peak_position.y.asin().to_degrees(),
            strict_domain_peak_longitude_deg: peak_position.z.atan2(peak_position.x).to_degrees(),
            strict_domain_land_area_km2: area(&land_cells),
            strict_domain_visible_cell_count: visible_cells.len(),
            strict_domain_visible_area_km2: area(&visible_cells),
            visible_component_count: visible_components.len(),
            largest_visible_component_cell_count: largest_visible.len(),
            largest_visible_component_area_km2: area(&largest_visible),
            largest_visible_component_hash: hash_cells(&largest_visible),
            largest_visible_component_cells: largest_visible,
            strict_domain_integration_cut_cells: count(&binding.fine_strict_owned_cells, &|cell| {
                surface.hydrology.was_lowered_by_integration(cell)
            }),
            strict_domain_breached_source_cells: count(&binding.fine_strict_owned_cells, &|cell| {
                surface.hydrology.integration_breached_source[cell]
            }),
            visible_integration_cut_cells: count(&visible_cells, &|cell| {
                surface.hydrology.was_lowered_by_integration(cell)
            }),
            visible_breached_source_cells: count(&visible_cells, &|cell| {
                surface.hydrology.integration_breached_source[cell]
            }),
        },
    };

    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.output, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {}", cli.output.display());
    println!(
        "strict={} ambiguous={} visible={} components={} largest={} peak={:.3}km",
        report.binding.fine_strict_owned_cell_count,
        report.binding.fine_ambiguous_association_cell_count,
        report.legacy_observation.strict_domain_visible_cell_count,
        report.legacy_observation.visible_component_count,
        report
            .legacy_observation
            .largest_visible_component_cell_count,
        report.legacy_observation.strict_domain_peak_elevation_km,
    );
    Ok(())
}

fn connected_components(tessellation: &Tessellation, mask: &[bool]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; mask.len()];
    let mut components = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            for &neighbor in tessellation.neighbors(cell) {
                if mask[neighbor] && !seen[neighbor] {
                    seen[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

fn edge_pair(id: BoundaryEdgeId) -> [usize; 2] {
    [id.cell_a, id.cell_b]
}

fn hash_cells(cells: &[usize]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &cell in cells {
        hash_bytes(&mut hash, &(cell as u64).to_le_bytes());
    }
    format!("{hash:016x}")
}

fn hash_edges(edges: &[BoundaryEdgeId]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for edge in edges {
        hash_bytes(&mut hash, &(edge.cell_a as u64).to_le_bytes());
        hash_bytes(&mut hash, &(edge.cell_b as u64).to_le_bytes());
    }
    format!("{hash:016x}")
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}
