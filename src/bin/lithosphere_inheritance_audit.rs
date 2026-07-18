//! Source-only audit of basement provinces and their exact plate-boundary relationships.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use hex3::world::{
    assess_plate_boundary_inheritance_v0, catalog_structural_source_belts,
    collect_convergent_fronts, collect_plate_boundaries, compile_structural_mountain,
    generate_lithosphere_inheritance_v0, generate_lithosphere_inheritance_with_history_seed_v0,
    query_boundary_inheritance_v0, select_primary_structural_source_belt, BasementProvinceV0,
    BoundaryEdgeId, BoundaryInheritanceContactKindV0, BoundaryKind, ConvergentFrontEdge,
    InheritedStructureEdgeV0, InheritedStructureIncidenceKindV0, InheritedStructureIncidenceV0,
    InheritedStructureKindV0, InheritedStructureRelationshipKindV0,
    InheritedStructureRelationshipTopologyV0, InheritedStructureRelationshipV0,
    InheritedStructureSegmentEndRefV0, InheritedStructureSegmentV0,
    LithosphereHistoryEventProvenanceV0, LithosphereHistoryEventV0, LithosphereInheritanceConfigV0,
    LithosphereInheritanceV0, PlateBoundaryEdge, StructuralMountainGraph, StructuralRegime,
    StructuralSegment, VoronoiBackend, World, LITHOSPHERE_INHERITANCE_SEED_SALT,
    NUM_PLATES_DEFAULT, PLANET_RADIUS_KM,
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
        default_value = "docs/generated/lithosphere-inheritance-seed-12345-v2.json"
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
    history: HistorySummary,
    history_resample_counterfactual: HistoryResampleCounterfactual,
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
    tip_incidence_count: usize,
    multi_trace_incidence_count: usize,
    explicit_relationship_count: usize,
    total_length_km: f64,
    retained_payload_bytes: usize,
    segment_length_km: ScalarSummary,
}

#[derive(Serialize)]
struct HistorySummary {
    assembly_event_count: usize,
    paleorift_event_count: usize,
    per_kind: Vec<StructureKindSummary>,
    original_candidate_contact_length_km: f64,
    named_length_km: f64,
    /// All named support divided by original candidate-contact length. This is
    /// a scale ratio, not a subset fraction, because paleorifts are internal.
    named_length_to_candidate_contact_ratio: f64,
    /// Preserved suture support divided by the contact skeleton it was selected
    /// from. Unlike the all-named ratio, numerator is a subset of denominator.
    preserved_suture_fraction_of_candidate_contact_length: f64,
    /// Fraction of named length in segments at least 300 km long and containing
    /// at least four exact source edges.
    named_coherent_length_fraction: f64,
    /// Fraction of named length carried by one-edge segments.
    single_edge_named_length_fraction: f64,
}

#[derive(Serialize)]
struct StructureKindSummary {
    kind: InheritedStructureKindV0,
    source_edge_count: usize,
    segment_count: usize,
    length_km: f64,
    segment_length_km: ScalarSummary,
}

#[derive(Serialize)]
struct HistoryResampleCounterfactual {
    candidate_geometry_identical: bool,
    named_edge_jaccard: f64,
    baseline_named_length_km: f64,
    resampled_named_length_km: f64,
    named_boundary_contact_location_change_count: usize,
}

#[derive(Serialize)]
struct BoundaryKindSummary {
    kind: String,
    edge_count: usize,
    application_counts: BTreeMap<String, usize>,
    geology_counts: BTreeMap<String, usize>,
    application_geology_counts: BTreeMap<String, usize>,
    unrelated_count: usize,
    coincident_count: usize,
    vertex_contact_count: usize,
    multi_trace_incidence_contact_count: usize,
    explicit_relationship_contact_count: usize,
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
    multi_trace_incidence_contact_count: usize,
    explicit_relationship_contact_count: usize,
    contact_events: Vec<ParentContactEvent>,
}

#[derive(Serialize)]
struct ParentContactEvent {
    edge_id: [usize; 2],
    midpoint_km: f32,
    kind: String,
    shared_vertices: Vec<u32>,
    structure_segment_ids: Vec<u32>,
    structure_kinds: Vec<InheritedStructureKindV0>,
    history_event_ids: Vec<u32>,
    geometric_incidence_ids: Vec<u32>,
    structure_relationship_ids: Vec<u32>,
    structure_relationship_kinds: Vec<InheritedStructureRelationshipKindV0>,
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
    let basement_seed = cli.seed.wrapping_add(LITHOSPHERE_INHERITANCE_SEED_SALT);
    let inheritance = generate_lithosphere_inheritance_v0(
        basement_seed,
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
    let history = history_summary(&inheritance);
    let resampled = generate_lithosphere_inheritance_with_history_seed_v0(
        basement_seed,
        cli.seed ^ 0x7265_7361_6d70_6c65,
        &world.tessellation,
        world.crust.as_ref().expect("crust generated"),
        config,
    )?;
    let history_resample_counterfactual =
        history_resample_counterfactual(&world, &inheritance, &resampled, &boundaries)?;
    let plate_boundaries = boundary_summaries(&world, &inheritance, &boundaries)?;
    let selected_collision_parent =
        selected_parent_summary(&world, &inheritance, &fronts.edges, source.id, parent)?;
    let report = Report {
        schema: "hex3-lithosphere-inheritance-audit-v2",
        seed: cli.seed,
        requested_coarse_cells: cli.cells,
        elapsed_seconds: started.elapsed().as_secs_f32(),
        inheritance_generation_seconds,
        config,
        provinces,
        graph,
        history,
        history_resample_counterfactual,
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
        tip_incidence_count: inheritance
            .graph
            .incidences
            .iter()
            .filter(|incidence| incidence.kind == InheritedStructureIncidenceKindV0::Tip)
            .count(),
        multi_trace_incidence_count: inheritance
            .graph
            .incidences
            .iter()
            .filter(|incidence| incidence.kind == InheritedStructureIncidenceKindV0::MultiTrace)
            .count(),
        explicit_relationship_count: inheritance.graph.relationships.len(),
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

fn history_summary(inheritance: &LithosphereInheritanceV0) -> HistorySummary {
    let assembly_event_count = inheritance
        .history_events
        .iter()
        .filter(|event| {
            matches!(
                event.provenance,
                LithosphereHistoryEventProvenanceV0::TerraneAssembly { .. }
            )
        })
        .count();
    let paleorift_event_count = inheritance
        .history_events
        .iter()
        .filter(|event| {
            matches!(
                event.provenance,
                LithosphereHistoryEventProvenanceV0::Paleorift { .. }
            )
        })
        .count();
    let per_kind = [
        InheritedStructureKindV0::BasementContact,
        InheritedStructureKindV0::Suture,
        InheritedStructureKindV0::InheritedRift,
        InheritedStructureKindV0::TransferLink,
    ]
    .into_iter()
    .map(|kind| StructureKindSummary {
        kind,
        source_edge_count: inheritance
            .graph
            .edges
            .iter()
            .filter(|edge| edge.kind == kind)
            .count(),
        segment_count: inheritance
            .graph
            .segments
            .iter()
            .filter(|segment| segment.kind == kind)
            .count(),
        length_km: inheritance
            .graph
            .segments
            .iter()
            .filter(|segment| segment.kind == kind)
            .map(|segment| f64::from(segment.length_km))
            .sum(),
        segment_length_km: summarize(
            inheritance
                .graph
                .segments
                .iter()
                .filter(|segment| segment.kind == kind)
                .map(|segment| f64::from(segment.length_km)),
        ),
    })
    .collect();
    let original_candidate_contact_length_km = inheritance
        .graph
        .edges
        .iter()
        .filter(|edge| edge.provinces[0] != edge.provinces[1])
        .map(|edge| f64::from(edge.length_km))
        .sum();
    let named_segments: Vec<_> = inheritance
        .graph
        .segments
        .iter()
        .filter(|segment| segment.kind != InheritedStructureKindV0::BasementContact)
        .collect();
    let named_length_km: f64 = named_segments
        .iter()
        .map(|segment| f64::from(segment.length_km))
        .sum();
    let suture_length_km: f64 = named_segments
        .iter()
        .filter(|segment| segment.kind == InheritedStructureKindV0::Suture)
        .map(|segment| f64::from(segment.length_km))
        .sum();
    let coherent_named_length_km: f64 = named_segments
        .iter()
        .filter(|segment| segment.length_km >= 300.0 && segment.source_edges.len() >= 4)
        .map(|segment| f64::from(segment.length_km))
        .sum();
    let single_edge_named_length_km: f64 = named_segments
        .iter()
        .filter(|segment| segment.source_edges.len() == 1)
        .map(|segment| f64::from(segment.length_km))
        .sum();
    HistorySummary {
        assembly_event_count,
        paleorift_event_count,
        per_kind,
        original_candidate_contact_length_km,
        named_length_km,
        named_length_to_candidate_contact_ratio: ratio(
            named_length_km,
            original_candidate_contact_length_km,
        ),
        preserved_suture_fraction_of_candidate_contact_length: ratio(
            suture_length_km,
            original_candidate_contact_length_km,
        ),
        named_coherent_length_fraction: ratio(coherent_named_length_km, named_length_km),
        single_edge_named_length_fraction: ratio(single_edge_named_length_km, named_length_km),
    }
}

fn history_resample_counterfactual(
    world: &World,
    baseline: &LithosphereInheritanceV0,
    resampled: &LithosphereInheritanceV0,
    boundaries: &[PlateBoundaryEdge],
) -> Result<HistoryResampleCounterfactual, Box<dyn std::error::Error>> {
    let candidate_geometry = |inheritance: &LithosphereInheritanceV0| {
        inheritance
            .graph
            .edges
            .iter()
            .filter(|edge| edge.provinces[0] != edge.provinces[1])
            .map(|edge| {
                (
                    edge.id,
                    edge.cells,
                    edge.vertices,
                    edge.endpoints,
                    edge.length_km,
                    edge.provinces,
                )
            })
            .collect::<Vec<_>>()
    };
    let named_edges = |inheritance: &LithosphereInheritanceV0| {
        inheritance
            .graph
            .edges
            .iter()
            .filter(|edge| edge.kind != InheritedStructureKindV0::BasementContact)
            .map(|edge| edge.id)
            .collect::<BTreeSet<_>>()
    };
    let baseline_named_edges = named_edges(baseline);
    let resampled_named_edges = named_edges(resampled);
    let union_count = baseline_named_edges.union(&resampled_named_edges).count();
    let named_edge_jaccard = if union_count == 0 {
        1.0
    } else {
        baseline_named_edges
            .intersection(&resampled_named_edges)
            .count() as f64
            / union_count as f64
    };
    let named_length = |inheritance: &LithosphereInheritanceV0| {
        inheritance
            .graph
            .edges
            .iter()
            .filter(|edge| edge.kind != InheritedStructureKindV0::BasementContact)
            .map(|edge| f64::from(edge.length_km))
            .sum()
    };
    let mut named_boundary_contact_location_change_count = 0;
    for boundary in boundaries {
        let id = BoundaryEdgeId::new(boundary.cell_a, boundary.cell_b);
        let baseline_relationship =
            query_boundary_inheritance_v0(&world.tessellation, baseline, id)?;
        let resampled_relationship =
            query_boundary_inheritance_v0(&world.tessellation, resampled, id)?;
        if relationship_contacts_named_structure(
            baseline,
            &baseline_relationship.structure_segment_ids,
        ) != relationship_contacts_named_structure(
            resampled,
            &resampled_relationship.structure_segment_ids,
        ) {
            named_boundary_contact_location_change_count += 1;
        }
    }
    Ok(HistoryResampleCounterfactual {
        candidate_geometry_identical: baseline.cell_province == resampled.cell_province
            && baseline.provinces == resampled.provinces
            && candidate_geometry(baseline) == candidate_geometry(resampled),
        named_edge_jaccard,
        baseline_named_length_km: named_length(baseline),
        resampled_named_length_km: named_length(resampled),
        named_boundary_contact_location_change_count,
    })
}

fn relationship_contacts_named_structure(
    inheritance: &LithosphereInheritanceV0,
    segment_ids: &[u32],
) -> bool {
    segment_ids.iter().any(|id| {
        inheritance.graph.segments.iter().any(|segment| {
            segment.id == *id && segment.kind != InheritedStructureKindV0::BasementContact
        })
    })
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn retained_payload_bytes(inheritance: &LithosphereInheritanceV0) -> usize {
    std::mem::size_of::<LithosphereInheritanceV0>()
        + inheritance.cell_province.len() * std::mem::size_of::<u32>()
        + inheritance.provinces.len() * std::mem::size_of::<BasementProvinceV0>()
        + inheritance.history_events.len() * std::mem::size_of::<LithosphereHistoryEventV0>()
        + inheritance
            .history_events
            .iter()
            .map(|event| match &event.provenance {
                LithosphereHistoryEventProvenanceV0::TerraneAssembly {
                    component_a,
                    component_b,
                } => (component_a.len() + component_b.len()) * std::mem::size_of::<u32>(),
                LithosphereHistoryEventProvenanceV0::Paleorift { .. } => 0,
            })
            .sum::<usize>()
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
        + inheritance.graph.incidences.len() * std::mem::size_of::<InheritedStructureIncidenceV0>()
        + inheritance
            .graph
            .incidences
            .iter()
            .map(|incidence| incidence.incident_segments.len() * std::mem::size_of::<u32>())
            .sum::<usize>()
        + inheritance.graph.relationships.len()
            * std::mem::size_of::<InheritedStructureRelationshipV0>()
        + inheritance
            .graph
            .relationships
            .iter()
            .map(|relationship| match &relationship.topology {
                InheritedStructureRelationshipTopologyV0::Junction { ends } => {
                    ends.len() * std::mem::size_of::<InheritedStructureSegmentEndRefV0>()
                }
                _ => 0,
            })
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
        let assessments = members
            .iter()
            .zip(&relationships)
            .map(|(edge, relationship)| {
                assess_plate_boundary_inheritance_v0(edge, relationship, &inheritance.graph)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut application_counts = BTreeMap::new();
        let mut geology_counts = BTreeMap::new();
        let mut application_geology_counts = BTreeMap::new();
        for assessment in &assessments {
            *application_counts
                .entry(format!("{:?}", assessment.application))
                .or_insert(0) += 1;
            *geology_counts
                .entry(format!("{:?}", assessment.geology))
                .or_insert(0) += 1;
            *application_geology_counts
                .entry(format!(
                    "{:?}/{:?}",
                    assessment.application, assessment.geology
                ))
                .or_insert(0) += 1;
        }
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
            application_counts,
            geology_counts,
            application_geology_counts,
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
            multi_trace_incidence_contact_count: relationships
                .iter()
                .filter(|relation| {
                    relation.geometric_incidence_ids.iter().any(|id| {
                        inheritance.graph.incidences.iter().any(|incidence| {
                            incidence.id == *id
                                && incidence.kind == InheritedStructureIncidenceKindV0::MultiTrace
                        })
                    })
                })
                .count(),
            explicit_relationship_contact_count: relationships
                .iter()
                .filter(|relation| !relation.structure_relationship_ids.is_empty())
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
    let mut multi_trace_incidence_contact_count = 0;
    let mut explicit_relationship_contact_count = 0;
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
        multi_trace_incidence_contact_count +=
            usize::from(relationship.geometric_incidence_ids.iter().any(|id| {
                inheritance.graph.incidences.iter().any(|incidence| {
                    incidence.id == *id
                        && incidence.kind == InheritedStructureIncidenceKindV0::MultiTrace
                })
            }));
        explicit_relationship_contact_count +=
            usize::from(!relationship.structure_relationship_ids.is_empty());
        if relationship.kind != BoundaryInheritanceContactKindV0::Unrelated {
            let contacted_segments: Vec<_> = relationship
                .structure_segment_ids
                .iter()
                .filter_map(|id| {
                    inheritance
                        .graph
                        .segments
                        .iter()
                        .find(|segment| segment.id == *id)
                })
                .collect();
            events.push(ParentContactEvent {
                edge_id: edge_pair(*id),
                midpoint_km,
                kind: format!("{:?}", relationship.kind),
                shared_vertices: relationship.shared_vertices,
                structure_segment_ids: relationship.structure_segment_ids,
                structure_kinds: contacted_segments
                    .iter()
                    .map(|segment| segment.kind)
                    .collect(),
                history_event_ids: contacted_segments
                    .iter()
                    .filter_map(|segment| segment.history_event_id)
                    .collect(),
                geometric_incidence_ids: relationship.geometric_incidence_ids,
                structure_relationship_ids: relationship.structure_relationship_ids,
                structure_relationship_kinds: relationship.structure_relationship_kinds,
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
        multi_trace_incidence_contact_count,
        explicit_relationship_contact_count,
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
