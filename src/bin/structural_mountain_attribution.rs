//! Freeze and report the fixed Structural Mountain V0 target/source ancestry.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use hex3::world::{
    attribute_legacy_convergent_sources, collect_convergent_fronts, collect_plate_boundaries,
    compile_structural_mountain, elevation_to_km, filter_attributed_fronts,
    select_fixed_structural_target, BoundaryEdgeId, FineCacheMode, RunManifest,
    StructuralMountainGraph, VoronoiBackend, World, NUM_PLATES_DEFAULT,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Freeze the fixed Structural Mountain target and source-front ancestry")]
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
        default_value = "docs/generated/structural-mountain-seed-12345-attribution-v0.json"
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
    target: TargetReport,
    attribution: AttributionReport,
    graph: GraphReport,
}

#[derive(Serialize)]
struct TargetReport {
    anchor_cell: usize,
    peak_cell: usize,
    peak_elevation_km: f32,
    core_cell_count: usize,
    core_area_km2: f64,
    buffer_cell_count: usize,
    buffer_area_km2: f64,
    core_integration_cut_cells: usize,
    buffer_integration_cut_cells: usize,
    core_breached_source_cells: usize,
    buffer_breached_source_cells: usize,
    core_hash: String,
    buffer_hash: String,
    core_cells: Vec<usize>,
    catchment_buffer_cells: Vec<usize>,
    buffer_terminal_cells: Vec<usize>,
}

#[derive(Serialize)]
struct AttributionReport {
    coarse_read_cell_count: usize,
    geometric_source_edges: Vec<[usize; 2]>,
    co_seed_source_edges: Vec<[usize; 2]>,
    bridge_source_edges: Vec<[usize; 2]>,
    selected_source_edges: Vec<[usize; 2]>,
    diffuse_dependency_edges: Vec<[usize; 2]>,
}

#[derive(Serialize)]
struct GraphReport {
    readiness: String,
    segment_count: usize,
    node_count: usize,
    link_count: usize,
    omission_count: usize,
    source_opportunity_km2: f64,
    compiled_opportunity_km2: f64,
    omitted_opportunity_km2: f64,
    allocation_residual_km2: f64,
    accounting_residual_km2: f64,
    segments: Vec<SegmentReport>,
    links: Vec<LinkReport>,
    omissions: Vec<OmissionReport>,
}

#[derive(Serialize)]
struct SegmentReport {
    id: [usize; 2],
    source_edges: Vec<[usize; 2]>,
    episode_id: usize,
    regime: String,
    length_km: f32,
    declared_opportunity_km2: f64,
}

#[derive(Serialize)]
struct LinkReport {
    node_id: u32,
    segments: [[usize; 2]; 2],
    kind: String,
}

#[derive(Serialize)]
struct OmissionReport {
    reason: String,
    source_edges: Vec<[usize; 2]>,
    declared_opportunity_km2: f64,
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

    let fine = world.fine.as_ref().expect("fine world generated");
    let final_surface = fine.eroded.as_ref().expect("stage 4 generated");
    let target = select_fixed_structural_target(
        &fine.base.tessellation,
        &final_surface.elevation.values,
        &final_surface.hydrology,
    )?;
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
    let attribution = attribute_legacy_convergent_sources(
        &world.tessellation,
        &fine.base.tessellation,
        &fine.base.coarse_cell,
        world.plates.as_ref().unwrap(),
        world.crust.as_ref().unwrap(),
        world.features.as_ref().expect("features generated"),
        &boundaries,
        &fronts,
        &target.core_cells,
    )?;
    let selected_fronts = filter_attributed_fronts(&fronts, &attribution)?;
    let graph = compile_structural_mountain(&selected_fronts)?;

    let report = Report {
        schema: "hex3-structural-mountain-attribution-v0",
        seed: cli.seed,
        requested_coarse_cells: cli.cells,
        actual_fine_cells: fine.base.tessellation.num_cells(),
        elapsed_seconds: started.elapsed().as_secs_f32(),
        manifest: RunManifest::from_world(&world),
        target: TargetReport {
            anchor_cell: target.anchor_cell,
            peak_cell: target.peak_cell,
            peak_elevation_km: elevation_to_km(final_surface.elevation.values[target.peak_cell]),
            core_cell_count: target.core_cells.len(),
            core_area_km2: target.core_area_km2,
            buffer_cell_count: target.catchment_buffer_cells.len(),
            buffer_area_km2: target.catchment_buffer_area_km2,
            core_integration_cut_cells: target.provenance.core_integration_cut_cells,
            buffer_integration_cut_cells: target.provenance.buffer_integration_cut_cells,
            core_breached_source_cells: target.provenance.core_breached_source_cells,
            buffer_breached_source_cells: target.provenance.buffer_breached_source_cells,
            core_hash: hash_cells(&target.core_cells),
            buffer_hash: hash_cells(&target.catchment_buffer_cells),
            core_cells: target.core_cells,
            catchment_buffer_cells: target.catchment_buffer_cells,
            buffer_terminal_cells: target.buffer_terminal_cells,
        },
        attribution: AttributionReport {
            coarse_read_cell_count: attribution.coarse_read_cells.len(),
            geometric_source_edges: edge_pairs(&attribution.geometric_source_edges),
            co_seed_source_edges: edge_pairs(&attribution.co_seed_source_edges),
            bridge_source_edges: edge_pairs(&attribution.bridge_source_edges),
            selected_source_edges: edge_pairs(&attribution.selected_source_edges),
            diffuse_dependency_edges: edge_pairs(&attribution.diffuse_dependency_edges),
        },
        graph: graph_report(&graph),
    };
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.output, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {}", cli.output.display());
    println!(
        "core={} buffer={} sources={} diffuse={} segments={} readiness={}",
        report.target.core_cell_count,
        report.target.buffer_cell_count,
        report.attribution.selected_source_edges.len(),
        report.attribution.diffuse_dependency_edges.len(),
        report.graph.segment_count,
        report.graph.readiness,
    );
    Ok(())
}

fn edge_pairs(edges: &[BoundaryEdgeId]) -> Vec<[usize; 2]> {
    edges.iter().map(|id| [id.cell_a, id.cell_b]).collect()
}

fn hash_cells(cells: &[usize]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &cell in cells {
        for byte in (cell as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn graph_report(graph: &StructuralMountainGraph) -> GraphReport {
    GraphReport {
        readiness: format!("{:?}", graph.readiness),
        segment_count: graph.segments.len(),
        node_count: graph.nodes.len(),
        link_count: graph.links.len(),
        omission_count: graph.omissions.len(),
        source_opportunity_km2: graph.ledger.source_km2,
        compiled_opportunity_km2: graph.ledger.compiled_km2,
        omitted_opportunity_km2: graph.ledger.omitted_km2,
        allocation_residual_km2: graph.ledger.residual_km2,
        accounting_residual_km2: graph.ledger.accounting_residual_km2,
        segments: graph
            .segments
            .iter()
            .map(|segment| SegmentReport {
                id: [segment.id.cell_a, segment.id.cell_b],
                source_edges: edge_pairs(&segment.source_edges),
                episode_id: segment.episode_id,
                regime: format!("{:?}", segment.regime),
                length_km: segment.length_km,
                declared_opportunity_km2: segment.declared_opportunity_km2,
            })
            .collect(),
        links: graph
            .links
            .iter()
            .map(|link| LinkReport {
                node_id: link.node_id,
                segments: [
                    [link.segments[0].cell_a, link.segments[0].cell_b],
                    [link.segments[1].cell_a, link.segments[1].cell_b],
                ],
                kind: format!("{:?}", link.kind),
            })
            .collect(),
        omissions: graph
            .omissions
            .iter()
            .map(|omission| OmissionReport {
                reason: format!("{:?}", omission.reason),
                source_edges: edge_pairs(&omission.source_edges),
                declared_opportunity_km2: omission.declared_opportunity_km2,
            })
            .collect(),
    }
}
