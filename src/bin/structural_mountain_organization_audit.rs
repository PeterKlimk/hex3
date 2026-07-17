//! Audit whether the selected structural source contains defensible internal organization.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use glam::Vec3;
use hex3::world::{
    catalog_structural_source_belts, collect_convergent_fronts, collect_plate_boundaries,
    compile_structural_mountain, select_primary_structural_source_belt, BoundaryEdgeId,
    ConvergentFrontEdge, RunManifest, StructuralMountainGraph, StructuralRegime, StructuralSegment,
    VoronoiBackend, World, COLLISION_WIDTH, MAX_PLATE_SPEED_KM_PER_MYR, NUM_PLATES_DEFAULT,
    PLANET_RADIUS_KM, TRANSFORM_NORMAL_THRESHOLD,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Audit causal variation inside the selected mountain source")]
struct Cli {
    #[arg(long, default_value_t = 12_345)]
    seed: u64,
    #[arg(long, default_value_t = 100_000)]
    cells: usize,
    #[arg(
        long,
        default_value = "docs/generated/structural-mountain-seed-12345-organization-audit-v1.json"
    )]
    output: PathBuf,
    #[arg(
        long,
        default_value = "docs/generated/structural-mountain-seed-12345-organization-audit-v1.svg"
    )]
    svg_output: PathBuf,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    seed: u64,
    requested_coarse_cells: usize,
    elapsed_seconds: f32,
    manifest: RunManifest,
    selected_source_id: [usize; 2],
    selected_source_segment_count: usize,
    parent: ParentReport,
    scales: ScaleReport,
    raw_summary: ProfileSummary,
    one_width_summary: ProfileSummary,
    three_width_summary: ProfileSummary,
    finite_parent_opportunity_density_summary: ScalarSummary,
    persistent_finite_parent_opportunity_extrema: Vec<ExtremumReport>,
    persistent_convergence_extrema: Vec<ExtremumReport>,
    persistent_obliquity_extrema: Vec<ExtremumReport>,
    broad_bend_candidates: Vec<BendReport>,
    inherited_material: InheritedMaterialReport,
    samples: Vec<SampleReport>,
}

#[derive(Serialize)]
struct InheritedMaterialReport {
    /// Craton identity on the two plates, ordered by `parent.plate_pair`.
    craton_pair_count: usize,
    same_craton_length_fraction: f32,
    craton_pair_runs: Vec<CratonPairRunReport>,
    craton_transitions: Vec<CratonTransitionReport>,
    nearer_ocean_margin_raw_summary_km: ScalarSummary,
    nearer_ocean_margin_three_width_summary_km: ScalarSummary,
    persistent_nearer_ocean_margin_extrema: Vec<ExtremumReport>,
}

#[derive(Serialize)]
struct CratonPairRunReport {
    plate_craton_pair: [u32; 2],
    start_edge: [usize; 2],
    end_edge: [usize; 2],
    start_km: f32,
    end_km: f32,
    edge_count: usize,
}

#[derive(Serialize)]
struct CratonTransitionReport {
    from_plate_craton_pair: [u32; 2],
    to_plate_craton_pair: [u32; 2],
    between_edges: [[usize; 2]; 2],
    along_strike_km: f32,
}

#[derive(Serialize)]
struct ParentReport {
    id: [usize; 2],
    episode_id: usize,
    plate_pair: [usize; 2],
    regime: String,
    source_edge_count: usize,
    length_km: f32,
    declared_opportunity_km2: f64,
    episode_duration_myr_range: [f32; 2],
    episode_normal_displacement_km_range: [f32; 2],
    legacy_ineligible_edge_count: usize,
    legacy_ineligible_edge_runs: Vec<EdgeRunReport>,
}

#[derive(Serialize)]
struct EdgeRunReport {
    start_edge: [usize; 2],
    end_edge: [usize; 2],
    start_km: f32,
    end_km: f32,
    edge_count: usize,
}

#[derive(Serialize)]
struct ScaleReport {
    collision_width_km: f32,
    broad_scale_km: f32,
    mean_edge_length_km: f32,
    edges_per_collision_width: f32,
}

#[derive(Serialize)]
struct ProfileSummary {
    convergence_km_per_myr: ScalarSummary,
    absolute_shear_km_per_myr: ScalarSummary,
    obliquity_deg: ScalarSummary,
}

#[derive(Serialize)]
struct ScalarSummary {
    length_weighted_mean: f32,
    length_weighted_stddev: f32,
    min: f32,
    max: f32,
    range_over_mean: f32,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
enum ExtremumKind {
    Minimum,
    Maximum,
}

#[derive(Clone, Debug, Serialize)]
struct ExtremumReport {
    kind: ExtremumKind,
    edge_id: [usize; 2],
    along_strike_km: f32,
    latitude_deg: f32,
    longitude_deg: f32,
    broad_value: f32,
    one_width_value: f32,
    local_prominence: f32,
    prominence_over_broad_mean: f32,
}

#[derive(Clone, Debug, Serialize)]
struct BendReport {
    edge_id: [usize; 2],
    along_strike_km: f32,
    latitude_deg: f32,
    longitude_deg: f32,
    deflection_over_one_width_deg: f32,
    deflection_over_three_widths_deg: f32,
}

#[derive(Serialize)]
struct SampleReport {
    edge_id: [usize; 2],
    start_km: f32,
    midpoint_km: f32,
    end_km: f32,
    length_km: f32,
    convergence_km_per_myr: f32,
    shear_km_per_myr: f32,
    obliquity_deg: f32,
    legacy_eligible: bool,
    convergence_one_width: f32,
    convergence_three_widths: f32,
    obliquity_one_width_deg: f32,
    obliquity_three_widths_deg: f32,
    bend_one_width_deg: Option<f32>,
    bend_three_widths_deg: Option<f32>,
    plate_craton_pair: [u32; 2],
    plate_margin_distance_km: [f32; 2],
    nearer_ocean_margin_km: f32,
    nearer_ocean_margin_three_width_km: f32,
}

struct Profile {
    edges: Vec<ConvergentFrontEdge>,
    starts: Vec<f32>,
    midpoints: Vec<f32>,
    ends: Vec<f32>,
    lengths: Vec<f32>,
    vertices: Vec<Vec3>,
    convergence: Vec<f32>,
    shear: Vec<f32>,
    obliquity: Vec<f32>,
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
    let source = select_primary_structural_source_belt(&belts)?;
    let parent = dominant_collision_parent(&graph, &source.segment_ids)
        .ok_or("selected source has no collision parent")?;
    let profile = build_profile(parent, &fronts.edges)?;
    let crust = world.crust.as_ref().expect("crust generated");

    let one_width = COLLISION_WIDTH * PLANET_RADIUS_KM;
    let three_widths = 3.0 * one_width;
    let convergence_one = smooth(&profile, &profile.convergence, one_width);
    let convergence_broad = smooth(&profile, &profile.convergence, three_widths);
    // Edge shear sign belongs to the canonical cell-pair frame, not the
    // ordered parent direction. Magnitude is valid; along-strike signed
    // cancellation is not recoverable from this record.
    let absolute_shear: Vec<_> = profile.shear.iter().map(|value| value.abs()).collect();
    let shear_one = smooth(&profile, &absolute_shear, one_width);
    let shear_broad = smooth(&profile, &absolute_shear, three_widths);
    let obliquity_one = obliquity_profile(&convergence_one, &shear_one);
    let obliquity_broad = obliquity_profile(&convergence_broad, &shear_broad);
    let bend_one = bend_profile(&profile, one_width);
    let bend_broad = bend_profile(&profile, three_widths);
    let plate_craton_pairs: Vec<_> = profile
        .edges
        .iter()
        .map(|edge| plate_craton_pair(edge, crust, parent.plate_pair))
        .collect();
    let plate_margin_distances_km: Vec<_> = profile
        .edges
        .iter()
        .map(|edge| plate_margin_pair_km(edge, crust, parent.plate_pair))
        .collect();
    let nearer_ocean_margin_km: Vec<_> = plate_margin_distances_km
        .iter()
        .map(|distance| distance[0].min(distance[1]))
        .collect();
    let nearer_ocean_margin_broad = smooth(&profile, &nearer_ocean_margin_km, three_widths);
    let compiled_density: Vec<_> = parent
        .compiled_opportunity_km2
        .iter()
        .zip(&profile.lengths)
        .map(|(&opportunity, &length)| (opportunity / f64::from(length)) as f32)
        .collect();
    let compiled_density_one = smooth(&profile, &compiled_density, one_width);
    let compiled_density_broad = smooth(&profile, &compiled_density, three_widths);

    let persistent_convergence_extrema =
        persistent_extrema(&profile, &convergence_one, &convergence_broad, three_widths);
    let persistent_obliquity_extrema =
        persistent_extrema(&profile, &obliquity_one, &obliquity_broad, three_widths);
    let persistent_finite_parent_opportunity_extrema = persistent_extrema(
        &profile,
        &compiled_density_one,
        &compiled_density_broad,
        three_widths,
    );
    let broad_bend_candidates = persistent_bends(&profile, &bend_one, &bend_broad, three_widths);
    let legacy_threshold = TRANSFORM_NORMAL_THRESHOLD * MAX_PLATE_SPEED_KM_PER_MYR;
    let legacy_eligible: Vec<_> = profile
        .convergence
        .iter()
        .map(|&value| value.max(0.0) >= legacy_threshold)
        .collect();
    let duration_range = range(profile.edges.iter().map(|edge| edge.episode_duration_myr));
    let displacement_range = range(
        profile
            .edges
            .iter()
            .map(|edge| edge.episode_normal_displacement_km),
    );
    let mean_edge_length = profile.lengths.iter().sum::<f32>() / profile.lengths.len() as f32;

    let report = Report {
        schema: "hex3-structural-mountain-organization-audit-v1",
        seed: cli.seed,
        requested_coarse_cells: cli.cells,
        elapsed_seconds: started.elapsed().as_secs_f32(),
        manifest: RunManifest::from_world(&world),
        selected_source_id: edge_pair(source.id),
        selected_source_segment_count: source.segment_ids.len(),
        parent: ParentReport {
            id: edge_pair(parent.id),
            episode_id: parent.episode_id,
            plate_pair: parent.plate_pair,
            regime: format!("{:?}", parent.regime),
            source_edge_count: parent.source_edges.len(),
            length_km: parent.length_km,
            declared_opportunity_km2: parent.declared_opportunity_km2,
            episode_duration_myr_range: duration_range,
            episode_normal_displacement_km_range: displacement_range,
            legacy_ineligible_edge_count: legacy_eligible.iter().filter(|&&value| !value).count(),
            legacy_ineligible_edge_runs: false_runs(&profile, &legacy_eligible),
        },
        scales: ScaleReport {
            collision_width_km: one_width,
            broad_scale_km: three_widths,
            mean_edge_length_km: mean_edge_length,
            edges_per_collision_width: one_width / mean_edge_length,
        },
        raw_summary: profile_summary(
            &profile,
            &profile.convergence,
            &profile.shear,
            &profile.obliquity,
        ),
        one_width_summary: profile_summary(&profile, &convergence_one, &shear_one, &obliquity_one),
        three_width_summary: profile_summary(
            &profile,
            &convergence_broad,
            &shear_broad,
            &obliquity_broad,
        ),
        finite_parent_opportunity_density_summary: scalar_summary(
            &compiled_density_broad,
            &profile.lengths,
        ),
        persistent_finite_parent_opportunity_extrema,
        persistent_convergence_extrema,
        persistent_obliquity_extrema,
        broad_bend_candidates,
        inherited_material: InheritedMaterialReport {
            craton_pair_count: plate_craton_pairs
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            same_craton_length_fraction: profile
                .lengths
                .iter()
                .zip(&plate_craton_pairs)
                .filter(|(_, pair)| pair[0] == pair[1])
                .map(|(&length, _)| length)
                .sum::<f32>()
                / profile.lengths.iter().sum::<f32>(),
            craton_pair_runs: value_runs(&profile, &plate_craton_pairs),
            craton_transitions: value_transitions(&profile, &plate_craton_pairs),
            nearer_ocean_margin_raw_summary_km: scalar_summary(
                &nearer_ocean_margin_km,
                &profile.lengths,
            ),
            nearer_ocean_margin_three_width_summary_km: scalar_summary(
                &nearer_ocean_margin_broad,
                &profile.lengths,
            ),
            persistent_nearer_ocean_margin_extrema: persistent_extrema(
                &profile,
                &smooth(&profile, &nearer_ocean_margin_km, one_width),
                &nearer_ocean_margin_broad,
                three_widths,
            ),
        },
        samples: profile
            .edges
            .iter()
            .enumerate()
            .map(|(index, edge)| SampleReport {
                edge_id: edge_pair(edge.id),
                start_km: profile.starts[index],
                midpoint_km: profile.midpoints[index],
                end_km: profile.ends[index],
                length_km: profile.lengths[index],
                convergence_km_per_myr: profile.convergence[index],
                shear_km_per_myr: profile.shear[index],
                obliquity_deg: profile.obliquity[index],
                legacy_eligible: legacy_eligible[index],
                convergence_one_width: convergence_one[index],
                convergence_three_widths: convergence_broad[index],
                obliquity_one_width_deg: obliquity_one[index],
                obliquity_three_widths_deg: obliquity_broad[index],
                bend_one_width_deg: bend_one[index],
                bend_three_widths_deg: bend_broad[index],
                plate_craton_pair: plate_craton_pairs[index],
                plate_margin_distance_km: plate_margin_distances_km[index],
                nearer_ocean_margin_km: nearer_ocean_margin_km[index],
                nearer_ocean_margin_three_width_km: nearer_ocean_margin_broad[index],
            })
            .collect(),
    };
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.output, serde_json::to_vec_pretty(&report)?)?;
    if let Some(parent) = cli.svg_output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &cli.svg_output,
        render_svg(
            &profile,
            &convergence_broad,
            &obliquity_broad,
            &report.broad_bend_candidates,
            one_width,
            three_widths,
        ),
    )?;
    println!("wrote {}", cli.output.display());
    println!("wrote {}", cli.svg_output.display());
    println!(
        "parent_edges={} conv_extrema={} obliquity_extrema={} bends={} broad_conv={:.2}..{:.2} km/Myr",
        report.parent.source_edge_count,
        report.persistent_convergence_extrema.len(),
        report.persistent_obliquity_extrema.len(),
        report.broad_bend_candidates.len(),
        report.three_width_summary.convergence_km_per_myr.min,
        report.three_width_summary.convergence_km_per_myr.max,
    );
    Ok(())
}

fn plate_craton_pair(
    edge: &ConvergentFrontEdge,
    crust: &hex3::world::Crust,
    plate_pair: [usize; 2],
) -> [u32; 2] {
    let mut result = [u32::MAX; 2];
    for side in 0..2 {
        let position = plate_pair
            .iter()
            .position(|&plate| plate == edge.plates[side])
            .expect("parent edge plate belongs to parent plate pair");
        result[position] = crust.cell_craton[edge.cells[side]];
    }
    result
}

fn plate_margin_pair_km(
    edge: &ConvergentFrontEdge,
    crust: &hex3::world::Crust,
    plate_pair: [usize; 2],
) -> [f32; 2] {
    let mut result = [f32::NAN; 2];
    for side in 0..2 {
        let position = plate_pair
            .iter()
            .position(|&plate| plate == edge.plates[side])
            .expect("parent edge plate belongs to parent plate pair");
        result[position] = crust.margin_distance(edge.cells[side]) * PLANET_RADIUS_KM;
    }
    result
}

fn value_runs(profile: &Profile, values: &[[u32; 2]]) -> Vec<CratonPairRunReport> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < values.len() {
        let mut end = start;
        while end + 1 < values.len() && values[end + 1] == values[start] {
            end += 1;
        }
        runs.push(CratonPairRunReport {
            plate_craton_pair: values[start],
            start_edge: edge_pair(profile.edges[start].id),
            end_edge: edge_pair(profile.edges[end].id),
            start_km: profile.starts[start],
            end_km: profile.ends[end],
            edge_count: end - start + 1,
        });
        start = end + 1;
    }
    runs
}

fn value_transitions(profile: &Profile, values: &[[u32; 2]]) -> Vec<CratonTransitionReport> {
    (1..values.len())
        .filter(|&index| values[index] != values[index - 1])
        .map(|index| CratonTransitionReport {
            from_plate_craton_pair: values[index - 1],
            to_plate_craton_pair: values[index],
            between_edges: [
                edge_pair(profile.edges[index - 1].id),
                edge_pair(profile.edges[index].id),
            ],
            along_strike_km: profile.starts[index],
        })
        .collect()
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

fn build_profile(
    segment: &StructuralSegment,
    fronts: &[ConvergentFrontEdge],
) -> Result<Profile, &'static str> {
    let by_id: HashMap<_, _> = fronts.iter().map(|edge| (edge.id, edge)).collect();
    let edges: Vec<_> = segment
        .source_edges
        .iter()
        .map(|id| by_id.get(id).copied().cloned().ok_or("missing parent edge"))
        .collect::<Result<_, _>>()?;
    if segment.vertices_in_order.len() != edges.len() + 1 {
        return Err("parent vertex order length mismatch");
    }
    let mut vertex_positions = BTreeMap::new();
    for edge in &edges {
        for endpoint in 0..2 {
            vertex_positions.insert(edge.vertices[endpoint], edge.endpoints[endpoint]);
        }
    }
    let vertices = segment
        .vertices_in_order
        .iter()
        .map(|vertex| {
            vertex_positions
                .get(vertex)
                .copied()
                .ok_or("missing parent vertex")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut starts = Vec::with_capacity(edges.len());
    let mut midpoints = Vec::with_capacity(edges.len());
    let mut ends = Vec::with_capacity(edges.len());
    let mut lengths = Vec::with_capacity(edges.len());
    let mut distance = 0.0;
    for edge in &edges {
        starts.push(distance);
        midpoints.push(distance + 0.5 * edge.length_km);
        distance += edge.length_km;
        ends.push(distance);
        lengths.push(edge.length_km);
    }
    let convergence: Vec<_> = edges
        .iter()
        .map(|edge| edge.convergence_km_per_myr)
        .collect();
    let shear: Vec<_> = edges.iter().map(|edge| edge.shear_km_per_myr).collect();
    let obliquity = obliquity_profile(&convergence, &shear);
    Ok(Profile {
        edges,
        starts,
        midpoints,
        ends,
        lengths,
        vertices,
        convergence,
        shear,
        obliquity,
    })
}

fn smooth(profile: &Profile, values: &[f32], sigma_km: f32) -> Vec<f32> {
    profile
        .midpoints
        .iter()
        .map(|&center| {
            let mut weighted = 0.0;
            let mut total = 0.0;
            for index in 0..values.len() {
                let distance = (profile.midpoints[index] - center) / sigma_km;
                let weight = profile.lengths[index] * (-0.5 * distance * distance).exp();
                weighted += values[index] * weight;
                total += weight;
            }
            weighted / total
        })
        .collect()
}

fn obliquity_profile(convergence: &[f32], shear: &[f32]) -> Vec<f32> {
    convergence
        .iter()
        .zip(shear)
        .map(|(&normal, &tangent)| tangent.abs().atan2(normal.max(0.0)).to_degrees())
        .collect()
}

fn profile_summary(
    profile: &Profile,
    convergence: &[f32],
    shear: &[f32],
    obliquity: &[f32],
) -> ProfileSummary {
    ProfileSummary {
        convergence_km_per_myr: scalar_summary(convergence, &profile.lengths),
        absolute_shear_km_per_myr: scalar_summary(
            &shear.iter().map(|value| value.abs()).collect::<Vec<_>>(),
            &profile.lengths,
        ),
        obliquity_deg: scalar_summary(obliquity, &profile.lengths),
    }
}

fn scalar_summary(values: &[f32], lengths: &[f32]) -> ScalarSummary {
    let total = lengths.iter().sum::<f32>();
    let mean = values
        .iter()
        .zip(lengths)
        .map(|(&value, &length)| value * length)
        .sum::<f32>()
        / total;
    let variance = values
        .iter()
        .zip(lengths)
        .map(|(&value, &length)| (value - mean).powi(2) * length)
        .sum::<f32>()
        / total;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ScalarSummary {
        length_weighted_mean: mean,
        length_weighted_stddev: variance.sqrt(),
        min,
        max,
        range_over_mean: (max - min) / mean.abs().max(1e-6),
    }
}

fn persistent_extrema(
    profile: &Profile,
    one_width: &[f32],
    broad: &[f32],
    scale_km: f32,
) -> Vec<ExtremumReport> {
    let fine = extrema_indices(one_width);
    let broad_mean = scalar_summary(broad, &profile.lengths).length_weighted_mean;
    extrema_indices(broad)
        .into_iter()
        .filter(|&(index, kind)| {
            profile.midpoints[index] >= scale_km
                && profile.midpoints[index] <= profile.ends.last().copied().unwrap() - scale_km
                && fine.iter().any(|&(candidate, candidate_kind)| {
                    candidate_kind == kind
                        && (profile.midpoints[candidate] - profile.midpoints[index]).abs()
                            <= scale_km
                })
        })
        .map(|(index, kind)| {
            let prominence = local_prominence(profile, broad, index, kind, 2.0 * scale_km);
            ExtremumReport {
                kind,
                edge_id: edge_pair(profile.edges[index].id),
                along_strike_km: profile.midpoints[index],
                latitude_deg: profile.edges[index].midpoint.y.asin().to_degrees(),
                longitude_deg: profile.edges[index]
                    .midpoint
                    .z
                    .atan2(profile.edges[index].midpoint.x)
                    .to_degrees(),
                broad_value: broad[index],
                one_width_value: one_width[index],
                local_prominence: prominence,
                prominence_over_broad_mean: prominence / broad_mean.abs().max(1e-6),
            }
        })
        .collect()
}

fn extrema_indices(values: &[f32]) -> Vec<(usize, ExtremumKind)> {
    let mut extrema = Vec::new();
    for index in 1..values.len().saturating_sub(1) {
        if values[index] < values[index - 1] && values[index] <= values[index + 1] {
            extrema.push((index, ExtremumKind::Minimum));
        } else if values[index] > values[index - 1] && values[index] >= values[index + 1] {
            extrema.push((index, ExtremumKind::Maximum));
        }
    }
    extrema
}

fn local_prominence(
    profile: &Profile,
    values: &[f32],
    index: usize,
    kind: ExtremumKind,
    radius_km: f32,
) -> f32 {
    let center = profile.midpoints[index];
    let left = values
        .iter()
        .enumerate()
        .filter(|(candidate, _)| {
            profile.midpoints[*candidate] < center
                && center - profile.midpoints[*candidate] <= radius_km
        })
        .map(|(_, &value)| value);
    let right = values
        .iter()
        .enumerate()
        .filter(|(candidate, _)| {
            profile.midpoints[*candidate] > center
                && profile.midpoints[*candidate] - center <= radius_km
        })
        .map(|(_, &value)| value);
    match kind {
        ExtremumKind::Minimum => {
            left.fold(f32::NEG_INFINITY, f32::max)
                .min(right.fold(f32::NEG_INFINITY, f32::max))
                - values[index]
        }
        ExtremumKind::Maximum => {
            values[index]
                - left
                    .fold(f32::INFINITY, f32::min)
                    .max(right.fold(f32::INFINITY, f32::min))
        }
    }
    .max(0.0)
}

fn bend_profile(profile: &Profile, half_window_km: f32) -> Vec<Option<f32>> {
    let length = profile.ends.last().copied().unwrap_or(0.0);
    profile
        .midpoints
        .iter()
        .map(|&center_s| {
            if center_s < half_window_km || center_s + half_window_km > length {
                return None;
            }
            let before = point_at(profile, center_s - half_window_km);
            let center = point_at(profile, center_s);
            let after = point_at(profile, center_s + half_window_km);
            let toward_before = (before - center * before.dot(center)).normalize_or_zero();
            let toward_after = (after - center * after.dot(center)).normalize_or_zero();
            if toward_before == Vec3::ZERO || toward_after == Vec3::ZERO {
                None
            } else {
                Some(
                    (-toward_before)
                        .dot(toward_after)
                        .clamp(-1.0, 1.0)
                        .acos()
                        .to_degrees(),
                )
            }
        })
        .collect()
}

fn point_at(profile: &Profile, distance_km: f32) -> Vec3 {
    let distance = distance_km.clamp(0.0, profile.ends.last().copied().unwrap_or(0.0));
    let edge = profile
        .ends
        .partition_point(|&end| end < distance)
        .min(profile.lengths.len() - 1);
    let t = ((distance - profile.starts[edge]) / profile.lengths[edge]).clamp(0.0, 1.0);
    slerp(profile.vertices[edge], profile.vertices[edge + 1], t)
}

fn slerp(left: Vec3, right: Vec3, t: f32) -> Vec3 {
    let angle = left.dot(right).clamp(-1.0, 1.0).acos();
    if angle <= 1e-6 {
        return left.lerp(right, t).normalize_or_zero();
    }
    let sin = angle.sin();
    (left * ((1.0 - t) * angle).sin() / sin + right * (t * angle).sin() / sin).normalize()
}

fn persistent_bends(
    profile: &Profile,
    one_width: &[Option<f32>],
    broad: &[Option<f32>],
    separation_km: f32,
) -> Vec<BendReport> {
    let local_maxima = |values: &[Option<f32>]| {
        (1..values.len().saturating_sub(1))
            .filter(|&index| {
                let Some(value) = values[index] else {
                    return false;
                };
                values[index - 1].is_some_and(|left| value > left)
                    && values[index + 1].is_some_and(|right| value >= right)
            })
            .collect::<Vec<_>>()
    };
    let one_width_maxima = local_maxima(one_width);
    let mut candidates: Vec<_> = local_maxima(broad)
        .into_iter()
        .filter(|&index| {
            one_width_maxima.iter().any(|&fine| {
                (profile.midpoints[fine] - profile.midpoints[index]).abs() <= separation_km
            })
        })
        .map(|index| (index, broad[index].expect("local maximum is finite")))
        .collect();
    candidates.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| profile.edges[left.0].id.cmp(&profile.edges[right.0].id))
    });
    let mut selected = Vec::<(usize, f32)>::new();
    for candidate in candidates {
        if selected.iter().all(|&(index, _)| {
            (profile.midpoints[index] - profile.midpoints[candidate.0]).abs() >= separation_km
        }) {
            selected.push(candidate);
        }
    }
    let mut reports: Vec<_> = selected
        .into_iter()
        .map(|(index, broad_value)| BendReport {
            edge_id: edge_pair(profile.edges[index].id),
            along_strike_km: profile.midpoints[index],
            latitude_deg: profile.edges[index].midpoint.y.asin().to_degrees(),
            longitude_deg: profile.edges[index]
                .midpoint
                .z
                .atan2(profile.edges[index].midpoint.x)
                .to_degrees(),
            deflection_over_one_width_deg: one_width[index].unwrap_or(0.0),
            deflection_over_three_widths_deg: broad_value,
        })
        .collect();
    reports.sort_by(|left, right| left.along_strike_km.total_cmp(&right.along_strike_km));
    reports
}

fn false_runs(profile: &Profile, values: &[bool]) -> Vec<EdgeRunReport> {
    let mut runs = Vec::new();
    let mut index = 0;
    while index < values.len() {
        if values[index] {
            index += 1;
            continue;
        }
        let start = index;
        while index + 1 < values.len() && !values[index + 1] {
            index += 1;
        }
        runs.push(EdgeRunReport {
            start_edge: edge_pair(profile.edges[start].id),
            end_edge: edge_pair(profile.edges[index].id),
            start_km: profile.starts[start],
            end_km: profile.ends[index],
            edge_count: index - start + 1,
        });
        index += 1;
    }
    runs
}

fn range(values: impl Iterator<Item = f32>) -> [f32; 2] {
    values.fold([f32::INFINITY, f32::NEG_INFINITY], |range, value| {
        [range[0].min(value), range[1].max(value)]
    })
}

fn edge_pair(id: BoundaryEdgeId) -> [usize; 2] {
    [id.cell_a, id.cell_b]
}

fn render_svg(
    profile: &Profile,
    convergence: &[f32],
    obliquity: &[f32],
    bends: &[BendReport],
    one_width_km: f32,
    broad_scale_km: f32,
) -> String {
    let width = 1100.0f32;
    let map_top = 70.0;
    let map_height = 500.0;
    let chart_top = 640.0;
    let chart_height = 260.0;
    let left = 80.0;
    let right = 1020.0;
    let center = profile
        .vertices
        .iter()
        .copied()
        .fold(Vec3::ZERO, |sum, point| sum + point)
        .normalize();
    let east = Vec3::Y.cross(center).normalize_or_zero();
    let east = if east == Vec3::ZERO { Vec3::X } else { east };
    let north = center.cross(east).normalize();
    let projected: Vec<_> = profile
        .vertices
        .iter()
        .map(|point| (point.dot(east), point.dot(north)))
        .collect();
    let min_x = projected
        .iter()
        .map(|point| point.0)
        .fold(f32::INFINITY, f32::min);
    let max_x = projected
        .iter()
        .map(|point| point.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = projected
        .iter()
        .map(|point| point.1)
        .fold(f32::INFINITY, f32::min);
    let max_y = projected
        .iter()
        .map(|point| point.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let scale =
        ((right - left) / (max_x - min_x).max(1e-6)).min(map_height / (max_y - min_y).max(1e-6));
    let map_x = |value: f32| left + 0.5 * (right - left) + (value - 0.5 * (min_x + max_x)) * scale;
    let map_y = |value: f32| map_top + 0.5 * map_height - (value - 0.5 * (min_y + max_y)) * scale;
    let conv_min = convergence.iter().copied().fold(f32::INFINITY, f32::min);
    let conv_max = convergence
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let obl_min = obliquity.iter().copied().fold(f32::INFINITY, f32::min);
    let obl_max = obliquity.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total_length = profile.ends.last().copied().unwrap_or(1.0);
    let chart_x = |distance: f32| left + (right - left) * distance / total_length;
    let chart_y = |value: f32, min: f32, max: f32| {
        chart_top + chart_height
            - 25.0
            - (chart_height - 55.0) * (value - min) / (max - min).max(1e-6)
    };

    let mut svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="950" viewBox="0 0 {width} 950">
<rect width="100%" height="100%" fill="#10151d"/>
<style>text{{font-family:system-ui,sans-serif;fill:#dce6f2}} .muted{{fill:#8fa3b8}} .grid{{stroke:#344354;stroke-width:1}}</style>
<text x="{left}" y="34" font-size="22">Structural source organization audit</text>
<text x="{left}" y="56" font-size="13" class="muted">plan view colored by 382 km-smoothed convergence; circles are scale-persistent bend maxima</text>
<rect x="{left}" y="{map_top}" width="{}" height="{map_height}" rx="8" fill="#18212c" stroke="#344354"/>
"##,
        right - left
    );
    for index in 0..profile.edges.len() {
        let t = (convergence[index] - conv_min) / (conv_max - conv_min).max(1e-6);
        let hue = 215.0 - 205.0 * t;
        svg.push_str(&format!(
            r##"<line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="hsl({hue:.1} 80% 60%)" stroke-width="7" stroke-linecap="round"/>
"##,
            map_x(projected[index].0),
            map_y(projected[index].1),
            map_x(projected[index + 1].0),
            map_y(projected[index + 1].1),
        ));
    }
    for (ordinal, bend) in bends.iter().enumerate() {
        let point = point_at(profile, bend.along_strike_km);
        svg.push_str(&format!(
            r##"<circle cx="{:.2}" cy="{:.2}" r="10" fill="#ffcf5a" stroke="#10151d" stroke-width="3"/><text x="{:.2}" y="{:.2}" font-size="12">B{}</text>
"##,
            map_x(point.dot(east)),
            map_y(point.dot(north)),
            map_x(point.dot(east)) + 13.0,
            map_y(point.dot(north)) + 4.0,
            ordinal + 1,
        ));
    }
    svg.push_str(&format!(
        r##"<text x="{left}" y="615" font-size="13" class="muted">collision width = {one_width_km:.0} km · broad scale = {broad_scale_km:.0} km · parent length = {total_length:.0} km</text>
<rect x="{left}" y="{chart_top}" width="{}" height="{chart_height}" rx="8" fill="#18212c" stroke="#344354"/>
<line x1="{left}" y1="{}" x2="{right}" y2="{}" class="grid"/>
<line x1="{left}" y1="{}" x2="{right}" y2="{}" class="grid"/>
"##,
        right - left,
        chart_top + 0.5 * chart_height,
        chart_top + 0.5 * chart_height,
        chart_top + chart_height - 25.0,
        chart_top + chart_height - 25.0,
    ));
    for bend in bends {
        let x = chart_x(bend.along_strike_km);
        svg.push_str(&format!(
            r##"<line x1="{x:.2}" y1="{chart_top}" x2="{x:.2}" y2="{}" stroke="#ffcf5a" stroke-width="1.5" stroke-dasharray="5 5" opacity="0.7"/>
"##,
            chart_top + chart_height
        ));
    }
    let polyline = |values: &[f32], min: f32, max: f32| {
        values
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                format!(
                    "{:.2},{:.2}",
                    chart_x(profile.midpoints[index]),
                    chart_y(value, min, max)
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    svg.push_str(&format!(
        r##"<polyline points="{}" fill="none" stroke="#ff6b5f" stroke-width="3"/><polyline points="{}" fill="none" stroke="#63c7ff" stroke-width="3"/>
<text x="{}" y="{}" font-size="13" fill="#ff6b5f">convergence {:.1}–{:.1} km/Myr</text>
<text x="{}" y="{}" font-size="13" fill="#63c7ff">obliquity {:.1}–{:.1}°</text>
<text x="{left}" y="925" font-size="12" class="muted">0 km</text><text x="{}" y="925" font-size="12" class="muted">{total_length:.0} km</text>
</svg>"##,
        polyline(convergence, conv_min, conv_max),
        polyline(obliquity, obl_min, obl_max),
        left + 12.0,
        chart_top + 22.0,
        conv_min,
        conv_max,
        left + 250.0,
        chart_top + 22.0,
        obl_min,
        obl_max,
        right - 55.0,
    ));
    svg
}
