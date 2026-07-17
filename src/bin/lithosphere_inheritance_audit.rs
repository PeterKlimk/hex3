//! Source-only audit of basement provinces and their exact plate-boundary relationships.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use hex3::world::{
    catalog_structural_source_belts, collect_convergent_fronts, collect_plate_boundaries,
    compile_structural_mountain, generate_lithosphere_inheritance_v0,
    query_boundary_inheritance_v0, select_primary_structural_source_belt, BasementProvinceV0,
    BoundaryEdgeId, BoundaryInheritanceContactKindV0, BoundaryKind, ConvergentFrontEdge,
    InheritedStructureEdgeV0, InheritedStructureNodeKindV0, InheritedStructureNodeV0,
    InheritedStructureSegmentV0, LithosphereInheritanceConfigV0, LithosphereInheritanceV0,
    PlateBoundaryEdge, StructuralMountainGraph, StructuralRegime, StructuralSegment,
    VoronoiBackend, World, LITHOSPHERE_INHERITANCE_SEED_SALT, NUM_PLATES_DEFAULT, PLANET_RADIUS_KM,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Audit source-only lithosphere inheritance against product boundaries")]
struct Cli {
    #[arg(long, default_value_t = 12_345)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    cells: usize,
    #[arg(
        long,
        default_value = "docs/generated/lithosphere-inheritance-seed-12345-v0.json"
    )]
    output: PathBuf,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    seed: u64,
    requested_coarse_cells: usize,
    elapsed_seconds: f32,
    inheritance_generation_seconds: f32,
    config: LithosphereInheritanceConfigV0,
    provinces: ProvinceSummary,
    graph: GraphSummary,
    plate_boundaries: Vec<BoundaryKindSummary>,
    selected_collision_parent: SelectedParentSummary,
}

#[derive(Serialize)]
struct ProvinceSummary {
    count: usize,
    per_craton: Vec<CratonProvinceSummary>,
    area_km2: ScalarSummary,
}

#[derive(Serialize)]
struct CratonProvinceSummary {
    craton_id: u32,
    province_count: usize,
    area_km2: f64,
}

#[derive(Serialize)]
struct GraphSummary {
    edge_count: usize,
    segment_count: usize,
    open_segment_count: usize,
    closed_segment_count: usize,
    tip_count: usize,
    junction_count: usize,
    total_length_km: f64,
    retained_payload_bytes: usize,
    segment_length_km: ScalarSummary,
}

#[derive(Serialize)]
struct BoundaryKindSummary {
    kind: String,
    edge_count: usize,
    unrelated_count: usize,
    coincident_count: usize,
    vertex_contact_count: usize,
    junction_contact_count: usize,
    contacted_length_km: f64,
    tangent_angle_deg: Option<ScalarSummary>,
}

#[derive(Serialize)]
struct SelectedParentSummary {
    source_id: [usize; 2],
    parent_id: [usize; 2],
    edge_count: usize,
    length_km: f32,
    unrelated_count: usize,
    coincident_count: usize,
    vertex_contact_count: usize,
    junction_contact_count: usize,
    contact_events: Vec<ParentContactEvent>,
}

#[derive(Serialize)]
struct ParentContactEvent {
    edge_id: [usize; 2],
    midpoint_km: f32,
    kind: String,
    shared_vertices: Vec<u32>,
    structure_segment_ids: Vec<u32>,
    at_structure_junction: bool,
    minimum_tangent_angle_deg: Option<f32>,
}

#[derive(Serialize)]
struct ScalarSummary {
    min: f64,
    mean: f64,
    median: f64,
    max: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();
    let started = Instant::now();
    let mut world = World::new_with_options(cli.seed, cli.cells, 1, VoronoiBackend::ConvexHull);
    world.generate_plates(NUM_PLATES_DEFAULT);
    world.generate_crust();
    world.generate_dynamics();
    world.generate_features();
    let config = LithosphereInheritanceConfigV0::default();
    let inheritance_started = Instant::now();
    let inheritance = generate_lithosphere_inheritance_v0(
        cli.seed.wrapping_add(LITHOSPHERE_INHERITANCE_SEED_SALT),
        &world.tessellation,
        world.crust.as_ref().expect("crust generated"),
        config,
    )?;
    let inheritance_generation_seconds = inheritance_started.elapsed().as_secs_f32();
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
    let structural_graph = compile_structural_mountain(&fronts)?;
    let belts = catalog_structural_source_belts(&structural_graph)?;
    let source = select_primary_structural_source_belt(&belts)?;
    let parent = dominant_collision_parent(&structural_graph, &source.segment_ids)
        .ok_or("selected source has no collision parent")?;

    let provinces = province_summary(&inheritance);
    let graph = graph_summary(&inheritance);
    let plate_boundaries = boundary_summaries(&world, &inheritance, &boundaries)?;
    let selected_collision_parent =
        selected_parent_summary(&world, &inheritance, &fronts.edges, source.id, parent)?;
    let report = Report {
        schema: "hex3-lithosphere-inheritance-audit-v0",
        seed: cli.seed,
        requested_coarse_cells: cli.cells,
        elapsed_seconds: started.elapsed().as_secs_f32(),
        inheritance_generation_seconds,
        config,
        provinces,
        graph,
        plate_boundaries,
        selected_collision_parent,
    };
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.output, serde_json::to_vec_pretty(&report)?)?;
    println!("wrote {}", cli.output.display());
    println!(
        "provinces={} structure_segments={} selected_contacts={}",
        report.provinces.count,
        report.graph.segment_count,
        report.selected_collision_parent.contact_events.len()
    );
    Ok(())
}

fn province_summary(inheritance: &LithosphereInheritanceV0) -> ProvinceSummary {
    let mut per_craton = BTreeMap::<u32, (usize, f64)>::new();
    for province in &inheritance.provinces {
        let entry = per_craton.entry(province.craton_id).or_default();
        entry.0 += 1;
        entry.1 += province.area_km2;
    }
    ProvinceSummary {
        count: inheritance.provinces.len(),
        per_craton: per_craton
            .into_iter()
            .map(
                |(craton_id, (province_count, area_km2))| CratonProvinceSummary {
                    craton_id,
                    province_count,
                    area_km2,
                },
            )
            .collect(),
        area_km2: summarize(
            inheritance
                .provinces
                .iter()
                .map(|province| province.area_km2),
        ),
    }
}

fn graph_summary(inheritance: &LithosphereInheritanceV0) -> GraphSummary {
    GraphSummary {
        edge_count: inheritance.graph.edges.len(),
        segment_count: inheritance.graph.segments.len(),
        open_segment_count: inheritance
            .graph
            .segments
            .iter()
            .filter(|segment| !segment.closed)
            .count(),
        closed_segment_count: inheritance
            .graph
            .segments
            .iter()
            .filter(|segment| segment.closed)
            .count(),
        tip_count: inheritance
            .graph
            .nodes
            .iter()
            .filter(|node| node.kind == InheritedStructureNodeKindV0::Tip)
            .count(),
        junction_count: inheritance
            .graph
            .nodes
            .iter()
            .filter(|node| node.kind == InheritedStructureNodeKindV0::Junction)
            .count(),
        total_length_km: inheritance.graph.total_length_km,
        retained_payload_bytes: retained_payload_bytes(inheritance),
        segment_length_km: summarize(
            inheritance
                .graph
                .segments
                .iter()
                .map(|segment| f64::from(segment.length_km)),
        ),
    }
}

fn retained_payload_bytes(inheritance: &LithosphereInheritanceV0) -> usize {
    std::mem::size_of::<LithosphereInheritanceV0>()
        + inheritance.cell_province.len() * std::mem::size_of::<u32>()
        + inheritance.provinces.len() * std::mem::size_of::<BasementProvinceV0>()
        + inheritance.graph.edges.len() * std::mem::size_of::<InheritedStructureEdgeV0>()
        + inheritance.graph.segments.len() * std::mem::size_of::<InheritedStructureSegmentV0>()
        + inheritance
            .graph
            .segments
            .iter()
            .map(|segment| {
                segment.source_edges.len() * std::mem::size_of::<BoundaryEdgeId>()
                    + segment.vertices_in_order.len() * std::mem::size_of::<u32>()
            })
            .sum::<usize>()
        + inheritance.graph.nodes.len() * std::mem::size_of::<InheritedStructureNodeV0>()
        + inheritance
            .graph
            .nodes
            .iter()
            .map(|node| node.incident_segments.len() * std::mem::size_of::<u32>())
            .sum::<usize>()
}

fn boundary_summaries(
    world: &World,
    inheritance: &LithosphereInheritanceV0,
    boundaries: &[PlateBoundaryEdge],
) -> Result<Vec<BoundaryKindSummary>, Box<dyn std::error::Error>> {
    [
        BoundaryKind::Convergent,
        BoundaryKind::Divergent,
        BoundaryKind::Transform,
    ]
    .into_iter()
    .map(|kind| {
        let members: Vec<_> = boundaries.iter().filter(|edge| edge.kind == kind).collect();
        let relationships = members
            .iter()
            .map(|edge| {
                query_boundary_inheritance_v0(
                    &world.tessellation,
                    inheritance,
                    BoundaryEdgeId::new(edge.cell_a, edge.cell_b),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let contacted_length_km = members
            .iter()
            .zip(&relationships)
            .filter(|(_, relation)| relation.kind != BoundaryInheritanceContactKindV0::Unrelated)
            .map(|(edge, _)| chord_to_km(edge.edge_length))
            .sum();
        let angles: Vec<_> = relationships
            .iter()
            .filter_map(|relationship| relationship.minimum_tangent_angle_deg.map(f64::from))
            .collect();
        Ok(BoundaryKindSummary {
            kind: format!("{kind:?}"),
            edge_count: members.len(),
            unrelated_count: relationships
                .iter()
                .filter(|relation| relation.kind == BoundaryInheritanceContactKindV0::Unrelated)
                .count(),
            coincident_count: relationships
                .iter()
                .filter(|relation| relation.kind == BoundaryInheritanceContactKindV0::Coincident)
                .count(),
            vertex_contact_count: relationships
                .iter()
                .filter(|relation| relation.kind == BoundaryInheritanceContactKindV0::VertexContact)
                .count(),
            junction_contact_count: relationships
                .iter()
                .filter(|relation| relation.at_structure_junction)
                .count(),
            contacted_length_km,
            tangent_angle_deg: (!angles.is_empty()).then(|| summarize(angles.into_iter())),
        })
    })
    .collect()
}

fn selected_parent_summary(
    world: &World,
    inheritance: &LithosphereInheritanceV0,
    fronts: &[ConvergentFrontEdge],
    source_id: BoundaryEdgeId,
    parent: &StructuralSegment,
) -> Result<SelectedParentSummary, Box<dyn std::error::Error>> {
    let by_id: BTreeMap<_, _> = fronts.iter().map(|edge| (edge.id, edge)).collect();
    let mut distance = 0.0;
    let mut events = Vec::new();
    let mut unrelated_count = 0;
    let mut coincident_count = 0;
    let mut vertex_contact_count = 0;
    let mut junction_contact_count = 0;
    for id in &parent.source_edges {
        let edge = by_id[id];
        let midpoint_km = distance + 0.5 * edge.length_km;
        distance += edge.length_km;
        let relationship = query_boundary_inheritance_v0(&world.tessellation, inheritance, *id)?;
        match relationship.kind {
            BoundaryInheritanceContactKindV0::Unrelated => unrelated_count += 1,
            BoundaryInheritanceContactKindV0::Coincident => coincident_count += 1,
            BoundaryInheritanceContactKindV0::VertexContact => vertex_contact_count += 1,
        }
        junction_contact_count += usize::from(relationship.at_structure_junction);
        if relationship.kind != BoundaryInheritanceContactKindV0::Unrelated {
            events.push(ParentContactEvent {
                edge_id: edge_pair(*id),
                midpoint_km,
                kind: format!("{:?}", relationship.kind),
                shared_vertices: relationship.shared_vertices,
                structure_segment_ids: relationship.structure_segment_ids,
                at_structure_junction: relationship.at_structure_junction,
                minimum_tangent_angle_deg: relationship.minimum_tangent_angle_deg,
            });
        }
    }
    Ok(SelectedParentSummary {
        source_id: edge_pair(source_id),
        parent_id: edge_pair(parent.id),
        edge_count: parent.source_edges.len(),
        length_km: parent.length_km,
        unrelated_count,
        coincident_count,
        vertex_contact_count,
        junction_contact_count,
        contact_events: events,
    })
}

fn dominant_collision_parent<'a>(
    graph: &'a StructuralMountainGraph,
    ids: &[BoundaryEdgeId],
) -> Option<&'a StructuralSegment> {
    graph
        .segments
        .iter()
        .filter(|segment| {
            ids.contains(&segment.id) && segment.regime == StructuralRegime::Collision
        })
        .max_by(|left, right| {
            left.declared_opportunity_km2
                .total_cmp(&right.declared_opportunity_km2)
                .then_with(|| right.id.cmp(&left.id))
        })
}

fn summarize(values: impl Iterator<Item = f64>) -> ScalarSummary {
    let mut values: Vec<_> = values.collect();
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    ScalarSummary {
        min: values.first().copied().unwrap_or(0.0),
        mean,
        median: values
            .get(values.len().saturating_sub(1) / 2)
            .copied()
            .unwrap_or(0.0),
        max: values.last().copied().unwrap_or(0.0),
    }
}

fn chord_to_km(chord: f32) -> f64 {
    f64::from(2.0 * (0.5 * chord).clamp(0.0, 1.0).asin() * PLANET_RADIUS_KM)
}

fn edge_pair(id: BoundaryEdgeId) -> [usize; 2] {
    [id.cell_a, id.cell_b]
}
