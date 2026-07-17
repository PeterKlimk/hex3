//! Select a Structural Mountain target before observing generated terrain.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use hex3::world::{
    catalog_structural_source_belts, collect_convergent_fronts, collect_plate_boundaries,
    compile_structural_mountain, ranked_continental_source_belts,
    select_primary_structural_source_belt, BoundaryEdgeId, RunManifest, StructuralMountainGraph,
    StructuralSegment, StructuralSourceBelt, VoronoiBackend, World, NUM_PLATES_DEFAULT,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Select a source-first structural mountain target")]
struct Cli {
    #[arg(long, default_value_t = 12_345)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    cells: usize,
    #[arg(
        long,
        default_value = "docs/generated/structural-mountain-seed-12345-source-target-v0.json"
    )]
    output: PathBuf,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    seed: u64,
    requested_coarse_cells: usize,
    elapsed_seconds: f32,
    manifest: RunManifest,
    convergent_front_edge_count: usize,
    global_graph: GlobalGraphReport,
    belt_count: usize,
    continental_relief_capable_belt_count: usize,
    selected: BeltReport,
    selected_segments: Vec<SegmentReport>,
    eligible_ranking: Vec<BeltReport>,
}

#[derive(Serialize)]
struct GlobalGraphReport {
    segment_count: usize,
    link_count: usize,
    omission_count: usize,
    readiness: String,
    source_opportunity_km2: f64,
    accounting_residual_km2: f64,
}

#[derive(Serialize)]
struct BeltReport {
    id: [usize; 2],
    episode_id: usize,
    plate_pair: [usize; 2],
    segment_ids: Vec<[usize; 2]>,
    source_edges: Vec<[usize; 2]>,
    segment_count: usize,
    source_edge_count: usize,
    length_km: f32,
    declared_opportunity_km2: f64,
    collision_segment_count: usize,
    subduction_segment_count: usize,
    continental_relief_capable: bool,
    readiness: String,
}

#[derive(Serialize)]
struct SegmentReport {
    id: [usize; 2],
    episode_id: usize,
    plate_pair: [usize; 2],
    regime: String,
    crust_on_plate_pair: [String; 2],
    subducting_plate: Option<usize>,
    receiving_plate: Option<usize>,
    source_edge_count: usize,
    length_km: f32,
    declared_opportunity_km2: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    let started = Instant::now();
    let mut world = World::new_with_options(cli.seed, cli.cells, 1, VoronoiBackend::ConvexHull);
    world.generate_all(NUM_PLATES_DEFAULT);

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
    let selected = select_primary_structural_source_belt(&belts)?;
    let ranked = ranked_continental_source_belts(&belts);

    let report = Report {
        schema: "hex3-structural-mountain-source-target-v0",
        seed: cli.seed,
        requested_coarse_cells: cli.cells,
        elapsed_seconds: started.elapsed().as_secs_f32(),
        manifest: RunManifest::from_world(&world),
        convergent_front_edge_count: fronts.edges.len(),
        global_graph: graph_report(&graph),
        belt_count: belts.len(),
        continental_relief_capable_belt_count: ranked.len(),
        selected: belt_report(selected),
        selected_segments: selected
            .segment_ids
            .iter()
            .map(|id| {
                graph
                    .segments
                    .iter()
                    .find(|segment| segment.id == *id)
                    .expect("selected segment belongs to compiled graph")
            })
            .map(segment_report)
            .collect(),
        eligible_ranking: ranked.into_iter().map(belt_report).collect(),
    };
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.output, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {}", cli.output.display());
    println!(
        "fronts={} belts={} eligible={} selected={:?} segments={} edges={} readiness={}",
        report.convergent_front_edge_count,
        report.belt_count,
        report.continental_relief_capable_belt_count,
        report.selected.id,
        report.selected.segment_count,
        report.selected.source_edge_count,
        report.selected.readiness,
    );
    Ok(())
}

fn graph_report(graph: &StructuralMountainGraph) -> GlobalGraphReport {
    GlobalGraphReport {
        segment_count: graph.segments.len(),
        link_count: graph.links.len(),
        omission_count: graph.omissions.len(),
        readiness: format!("{:?}", graph.readiness),
        source_opportunity_km2: graph.ledger.source_km2,
        accounting_residual_km2: graph.ledger.accounting_residual_km2,
    }
}

fn belt_report(belt: &StructuralSourceBelt) -> BeltReport {
    BeltReport {
        id: edge_pair(belt.id),
        episode_id: belt.episode_id,
        plate_pair: belt.plate_pair,
        segment_ids: belt.segment_ids.iter().copied().map(edge_pair).collect(),
        source_edges: belt.source_edges.iter().copied().map(edge_pair).collect(),
        segment_count: belt.segment_ids.len(),
        source_edge_count: belt.source_edges.len(),
        length_km: belt.length_km,
        declared_opportunity_km2: belt.declared_opportunity_km2,
        collision_segment_count: belt.collision_segment_count,
        subduction_segment_count: belt.subduction_segment_count,
        continental_relief_capable: belt.continental_relief_capable,
        readiness: format!("{:?}", belt.readiness),
    }
}

fn segment_report(segment: &StructuralSegment) -> SegmentReport {
    SegmentReport {
        id: edge_pair(segment.id),
        episode_id: segment.episode_id,
        plate_pair: segment.plate_pair,
        regime: format!("{:?}", segment.regime),
        crust_on_plate_pair: segment
            .crust_on_plate_pair
            .map(|crust| format!("{crust:?}")),
        subducting_plate: segment.subducting_plate,
        receiving_plate: segment.receiving_plate,
        source_edge_count: segment.source_edges.len(),
        length_km: segment.length_km,
        declared_opportunity_km2: segment.declared_opportunity_km2,
    }
}

fn edge_pair(id: BoundaryEdgeId) -> [usize; 2] {
    [id.cell_a, id.cell_b]
}
