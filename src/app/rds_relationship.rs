//! Lightweight relationship readout for the fixed RDS0 terrain discriminator.
//!
//! This module observes existing source, terrain, and product hydrology. It
//! does not alter physical state, introduce a terrain mechanism, or claim that
//! a graph boundary between drainage destinations is a geomorphic divide.

use std::collections::{BTreeMap, BTreeSet};

use glam::Vec3;
use serde::Serialize;

use hex3::world::diagnostics::measure_components;
use hex3::world::{
    elevation_to_km, solid_angle_to_km2, FineSurface, Hydrology, RegionalDeformationRasterV0,
    RiverNetwork, RiverThresholdPolicy, Tessellation, WaterBodySemantics, PLANET_RADIUS_KM,
};

const SOURCE_EPSILON: f64 = 1.0e-15;
const HIGHLAND_THRESHOLD_KM: f32 = 1.5;
const HIGHLAND_RECORD_LIMIT: usize = 32;
const TRUNK_RECORD_LIMIT: usize = 5;

#[derive(Clone, Debug)]
pub struct RdsScheduleCompressionV0 {
    pub mean_source_density_per_myr: Vec<f64>,
    pub active_frame_count: Vec<u8>,
    pub dominant_lineage: Vec<Option<u8>>,
    pub dominant_lineage_share: Vec<f32>,
    pub axial_fabric: Vec<Vec3>,
}

impl RdsScheduleCompressionV0 {
    fn cell_count(&self) -> usize {
        self.mean_source_density_per_myr.len()
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RdsRelationshipAnalysisV0 {
    pub summary: RdsRelationshipSummaryV0,
    #[serde(skip)]
    pub source_topology_colors: Vec<Vec3>,
    #[serde(skip)]
    pub catchment_divide_basin_colors: Vec<Vec3>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RdsRelationshipSummaryV0 {
    pub schema: &'static str,
    pub status: &'static str,
    pub highland_definition: &'static str,
    pub endpoint_definition: &'static str,
    pub catchment_boundary_semantics: &'static str,
    pub rigorous_saddle_status: &'static str,
    pub support: SupportSummaryV0,
    pub source: SourceSummaryV0,
    pub highlands: HighlandSummaryV0,
    pub catchments: CatchmentSummaryV0,
    pub trunks: TrunkSummaryV0,
    pub depressions: DepressionSummaryV0,
    pub diagnostic_legend: DiagnosticLegendV0,
}

#[derive(Clone, Debug, Serialize)]
pub struct SupportSummaryV0 {
    pub cell_count: usize,
    pub area_km2: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SourceSummaryV0 {
    pub active_cell_count: usize,
    pub active_support_cell_count: usize,
    pub total_rate_km2_per_myr: f64,
    pub support_rate_km2_per_myr: f64,
    pub support_rate_fraction: f64,
    pub rate_in_highlands_fraction: f64,
    pub rate_in_depressions_fraction: f64,
    pub rate_assigned_to_terminal_mouth_fraction: f64,
    pub largest_terminal_mouth_source_share: f64,
    pub effective_source_catchment_count: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HighlandSummaryV0 {
    pub threshold_km: f32,
    pub component_count: usize,
    pub recorded_largest_component_count: usize,
    pub source_intersecting_component_count: usize,
    pub catchment_spanning_component_count: usize,
    pub major_river_intersecting_component_count: usize,
    pub depression_intersecting_component_count: usize,
    pub records: Vec<HighlandRecordV0>,
}

#[derive(Clone, Debug, Serialize)]
pub struct HighlandRecordV0 {
    pub rank_by_area: usize,
    pub cell_count: usize,
    pub area_km2: f32,
    pub length_km: f32,
    pub mean_width_km: f32,
    pub elongation: f32,
    pub two_sweep_end_cells: [usize; 2],
    pub two_sweep_end_elevation_km: [f32; 2],
    pub source_rate_km2_per_myr: f64,
    pub dominant_lineage: Option<u8>,
    pub dominant_lineage_share: f64,
    pub distinct_terminal_mouth_catchments: usize,
    pub dominant_catchment_area_share: f64,
    pub catchment_boundary_proxy_cells: usize,
    pub major_river_cells: usize,
    pub depression_cells: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CatchmentSummaryV0 {
    pub policy: RiverThresholdPolicy,
    pub effective_minimum_catchment_km2: Option<f32>,
    pub terminal_mouth_count: usize,
    pub support_assigned_cell_count: usize,
    pub support_terminal_mouth_count: usize,
    pub boundary_proxy_cell_count: usize,
    pub boundary_proxy_area_km2: f64,
    pub boundary_proxy_support_fraction: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrunkSummaryV0 {
    pub selection_rule: &'static str,
    pub support_intersecting_trunk_count: usize,
    pub recorded_trunk_count: usize,
    pub records: Vec<TrunkRecordV0>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrunkRecordV0 {
    pub mouth_cell: usize,
    pub head_cell: usize,
    pub strahler_order_at_mouth: u8,
    pub mouth_discharge_equivalent_km2: f32,
    pub trunk_cell_count: usize,
    pub support_cell_count: usize,
    pub support_length_km: f64,
    pub support_boundary_crossings: usize,
    pub source_rate_km2_per_myr: f64,
    pub source_fabric_segment_count: usize,
    pub source_fabric_mean_acute_angle_deg: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DepressionSummaryV0 {
    pub support_intersecting_count: usize,
    pub source_intersecting_count: usize,
    pub highland_intersecting_count: usize,
    pub records: Vec<DepressionRecordV0>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DepressionRecordV0 {
    pub basin_id: usize,
    pub support_cell_count: usize,
    pub support_area_km2: f64,
    pub highland_cell_count: usize,
    pub source_active_cell_count: usize,
    pub source_rate_km2_per_myr: f64,
    pub boundary_source_density_per_myr: f64,
    pub interior_source_density_per_myr: f64,
    pub boundary_to_interior_source_ratio: Option<f64>,
    pub bottom_elevation_km: f32,
    pub spill_elevation_km: f32,
    pub sill_relief_km: f32,
    pub water_level_km: f32,
    pub has_water: bool,
    pub overflowing: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DiagnosticLegendV0 {
    pub source_topology: &'static str,
    pub catchment_divide_basin: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RdsRelationshipErrorV0 {
    EmptySchedule,
    LengthMismatch(&'static str),
    InvalidLineage(u8),
    NonFiniteSource,
    TooManyFrames,
}

impl std::fmt::Display for RdsRelationshipErrorV0 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RdsRelationshipErrorV0 {}

/// Collapse chronological source rasters before their sparse provenance is
/// released for the long terrain solve.
pub fn compress_rds0_schedule_v0(
    frames: &[&RegionalDeformationRasterV0],
) -> Result<RdsScheduleCompressionV0, RdsRelationshipErrorV0> {
    let first = frames
        .first()
        .ok_or(RdsRelationshipErrorV0::EmptySchedule)?;
    if frames.len() > usize::from(u8::MAX) {
        return Err(RdsRelationshipErrorV0::TooManyFrames);
    }
    let n = first.rate_density_per_myr.len();
    let mut density_sum = vec![0.0; n];
    let mut active_frame_count = vec![0; n];
    let mut lineage_sum = vec![[0.0; 3]; n];
    let mut fabric_sum = vec![Vec3::ZERO; n];
    for frame in frames {
        if frame.rate_density_per_myr.len() != n
            || frame.provenance.len() != n
            || frame.axial_fabric.len() != n
        {
            return Err(RdsRelationshipErrorV0::LengthMismatch("schedule raster"));
        }
        for cell in 0..n {
            let density = frame.rate_density_per_myr[cell];
            if !density.is_finite() || density < 0.0 || !frame.axial_fabric[cell].is_finite() {
                return Err(RdsRelationshipErrorV0::NonFiniteSource);
            }
            density_sum[cell] += density;
            active_frame_count[cell] += u8::from(density > SOURCE_EPSILON);
            fabric_sum[cell] += frame.axial_fabric[cell] * density as f32;
            for contribution in &frame.provenance[cell] {
                let lineage = usize::from(contribution.element_id.lineage);
                if contribution.element_id.lineage == u8::MAX {
                    // Static-control provenance deliberately has no temporal
                    // lineage identity. Its density/fabric remain observable.
                    continue;
                }
                if lineage >= 3 {
                    return Err(RdsRelationshipErrorV0::InvalidLineage(
                        contribution.element_id.lineage,
                    ));
                }
                let rate = contribution.rate_density_per_myr;
                if !rate.is_finite() || rate < 0.0 {
                    return Err(RdsRelationshipErrorV0::NonFiniteSource);
                }
                lineage_sum[cell][lineage] += rate;
            }
        }
    }
    let divisor = frames.len() as f64;
    let mean_source_density_per_myr = density_sum.iter().map(|value| value / divisor).collect();
    let mut dominant_lineage = Vec::with_capacity(n);
    let mut dominant_lineage_share = Vec::with_capacity(n);
    for sums in lineage_sum {
        let total: f64 = sums.iter().sum();
        if total <= SOURCE_EPSILON {
            dominant_lineage.push(None);
            dominant_lineage_share.push(0.0);
        } else {
            let winner = (0..3)
                .max_by(|&a, &b| sums[a].total_cmp(&sums[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            dominant_lineage.push(Some(winner as u8));
            dominant_lineage_share.push((sums[winner] / total) as f32);
        }
    }
    let axial_fabric = fabric_sum
        .into_iter()
        .map(|fabric| fabric.try_normalize().unwrap_or(Vec3::ZERO))
        .collect();
    Ok(RdsScheduleCompressionV0 {
        mean_source_density_per_myr,
        active_frame_count,
        dominant_lineage,
        dominant_lineage_share,
        axial_fabric,
    })
}

pub fn analyze_rds0_relationships_v0(
    tessellation: &Tessellation,
    surface: &FineSurface,
    support: &[bool],
    source: &RdsScheduleCompressionV0,
) -> Result<RdsRelationshipAnalysisV0, RdsRelationshipErrorV0> {
    let n = tessellation.num_cells();
    if surface.elevation.values.len() != n
        || surface.hydrology.drainage_dir.len() != n
        || support.len() != n
        || source.cell_count() != n
        || source.active_frame_count.len() != n
        || source.dominant_lineage.len() != n
        || source.dominant_lineage_share.len() != n
        || source.axial_fabric.len() != n
    {
        return Err(RdsRelationshipErrorV0::LengthMismatch("analysis domain"));
    }
    let area_km2: Vec<f64> = tessellation
        .cell_areas_ref()
        .iter()
        .map(|&area| f64::from(solid_angle_to_km2(area)))
        .collect();
    let highland_mask: Vec<bool> = (0..n)
        .map(|cell| {
            support[cell] && elevation_to_km(surface.elevation.values[cell]) > HIGHLAND_THRESHOLD_KM
        })
        .collect();
    let components = measure_components(tessellation, &highland_mask);

    let water = WaterBodySemantics::build(tessellation, &surface.hydrology);
    let river_policy = RiverThresholdPolicy::default();
    let network = RiverNetwork::build(tessellation, &surface.hydrology, &water, river_policy);
    let mut mouth_anchor = vec![None; n];
    for &mouth in &network.mouths {
        mouth_anchor[mouth] = Some(mouth);
    }
    let catchment_owner = downstream_anchor_owner(&surface.hydrology.drainage_dir, &mouth_anchor);
    let boundary_proxy = catchment_boundary_proxy(tessellation, support, &catchment_owner);

    let mut endpoint_mask = vec![false; n];
    let mut highland_records = Vec::new();
    let mut source_intersecting_components = 0;
    let mut catchment_spanning_components = 0;
    let mut river_intersecting_components = 0;
    let mut depression_intersecting_components = 0;
    for (rank, component) in components.iter().enumerate() {
        let ends = two_sweep_endpoints(tessellation, &component.cells);
        endpoint_mask[ends[0]] = true;
        endpoint_mask[ends[1]] = true;
        let source_rate = weighted_sum(&component.cells, &area_km2, |cell| {
            source.mean_source_density_per_myr[cell]
        });
        let mut catchment_area = BTreeMap::<usize, f64>::new();
        for &cell in &component.cells {
            if let Some(owner) = catchment_owner[cell] {
                *catchment_area.entry(owner).or_default() += area_km2[cell];
            }
        }
        let owned_area: f64 = catchment_area.values().sum();
        let dominant_catchment_area_share =
            catchment_area.values().copied().fold(0.0_f64, f64::max)
                / owned_area.max(f64::MIN_POSITIVE);
        let major_river_cells = component
            .cells
            .iter()
            .filter(|&&cell| network.major_cells[cell])
            .count();
        let depression_cells = component
            .cells
            .iter()
            .filter(|&&cell| surface.hydrology.basin_id[cell].is_some())
            .count();
        source_intersecting_components += usize::from(source_rate > SOURCE_EPSILON);
        catchment_spanning_components += usize::from(catchment_area.len() > 1);
        river_intersecting_components += usize::from(major_river_cells > 0);
        depression_intersecting_components += usize::from(depression_cells > 0);
        if rank < HIGHLAND_RECORD_LIMIT {
            let (lineage, share) = dominant_lineage_for_cells(&component.cells, &area_km2, source);
            highland_records.push(HighlandRecordV0 {
                rank_by_area: rank + 1,
                cell_count: component.cells.len(),
                area_km2: component.area_km2,
                length_km: component.length_km,
                mean_width_km: component.width_km,
                elongation: component.elongation(),
                two_sweep_end_cells: ends,
                two_sweep_end_elevation_km: ends
                    .map(|cell| elevation_to_km(surface.elevation.values[cell])),
                source_rate_km2_per_myr: source_rate,
                dominant_lineage: lineage,
                dominant_lineage_share: share,
                distinct_terminal_mouth_catchments: catchment_area.len(),
                dominant_catchment_area_share,
                catchment_boundary_proxy_cells: component
                    .cells
                    .iter()
                    .filter(|&&cell| boundary_proxy[cell])
                    .count(),
                major_river_cells,
                depression_cells,
            });
        }
    }

    let support_cells: Vec<usize> = (0..n).filter(|&cell| support[cell]).collect();
    let support_area = weighted_sum(&support_cells, &area_km2, |_| 1.0);
    let total_source_rate = (0..n)
        .map(|cell| source.mean_source_density_per_myr[cell] * area_km2[cell])
        .sum::<f64>();
    let support_source_rate = weighted_sum(&support_cells, &area_km2, |cell| {
        source.mean_source_density_per_myr[cell]
    });
    let highland_source_rate = weighted_sum(
        &support_cells
            .iter()
            .copied()
            .filter(|&cell| highland_mask[cell])
            .collect::<Vec<_>>(),
        &area_km2,
        |cell| source.mean_source_density_per_myr[cell],
    );
    let depression_source_rate = weighted_sum(
        &support_cells
            .iter()
            .copied()
            .filter(|&cell| surface.hydrology.basin_id[cell].is_some())
            .collect::<Vec<_>>(),
        &area_km2,
        |cell| source.mean_source_density_per_myr[cell],
    );
    let mut source_by_catchment = BTreeMap::<usize, f64>::new();
    for &cell in &support_cells {
        if let Some(owner) = catchment_owner[cell] {
            *source_by_catchment.entry(owner).or_default() +=
                source.mean_source_density_per_myr[cell] * area_km2[cell];
        }
    }
    let assigned_source: f64 = source_by_catchment.values().sum();

    let trunks = trunk_records(
        tessellation,
        &surface.hydrology,
        &network,
        support,
        source,
        &area_km2,
    );
    let support_intersecting_trunk_count = trunks.len();
    let recorded_trunks: Vec<_> = trunks.into_iter().take(TRUNK_RECORD_LIMIT).collect();
    let depressions = depression_records(
        tessellation,
        &surface.hydrology,
        support,
        &highland_mask,
        source,
        &area_km2,
    );
    let source_intersecting_depressions = depressions
        .iter()
        .filter(|record| record.source_rate_km2_per_myr > SOURCE_EPSILON)
        .count();
    let highland_intersecting_depressions = depressions
        .iter()
        .filter(|record| record.highland_cell_count > 0)
        .count();
    let boundary_area = support_cells
        .iter()
        .filter(|&&cell| boundary_proxy[cell])
        .map(|&cell| area_km2[cell])
        .sum::<f64>();
    let support_catchments: BTreeSet<usize> = support_cells
        .iter()
        .filter_map(|&cell| catchment_owner[cell])
        .collect();

    let summary = RdsRelationshipSummaryV0 {
        schema: "hex3.rds0-relationship.v0",
        status: "fixed-world observer; not a promotion metric or terrain mechanism",
        highland_definition: "connected target-land support above the established physical 1.5 km mountain threshold",
        endpoint_definition: "deterministic two-sweep farthest cell anchors; approximate finite component ends, not summits",
        catchment_boundary_semantics: "adjacent support cells draining to different represented terminal river mouths; a graph-label boundary proxy, not a geomorphic divide",
        rigorous_saddle_status: "omitted: the exact spherical peak-saddle hierarchy duplicates full f64 process-mesh geometry and is not justified for this first RAM-bounded falsifier",
        support: SupportSummaryV0 {
            cell_count: support_cells.len(),
            area_km2: support_area,
        },
        source: SourceSummaryV0 {
            active_cell_count: source.active_frame_count.iter().filter(|&&v| v > 0).count(),
            active_support_cell_count: support_cells
                .iter()
                .filter(|&&cell| source.active_frame_count[cell] > 0)
                .count(),
            total_rate_km2_per_myr: total_source_rate,
            support_rate_km2_per_myr: support_source_rate,
            support_rate_fraction: support_source_rate / total_source_rate.max(f64::MIN_POSITIVE),
            rate_in_highlands_fraction: highland_source_rate
                / support_source_rate.max(f64::MIN_POSITIVE),
            rate_in_depressions_fraction: depression_source_rate
                / support_source_rate.max(f64::MIN_POSITIVE),
            rate_assigned_to_terminal_mouth_fraction: assigned_source
                / support_source_rate.max(f64::MIN_POSITIVE),
            largest_terminal_mouth_source_share: source_by_catchment
                .values()
                .copied()
                .fold(0.0_f64, f64::max)
                / assigned_source.max(f64::MIN_POSITIVE),
            effective_source_catchment_count: effective_count(source_by_catchment.values().copied()),
        },
        highlands: HighlandSummaryV0 {
            threshold_km: HIGHLAND_THRESHOLD_KM,
            component_count: components.len(),
            recorded_largest_component_count: highland_records.len(),
            source_intersecting_component_count: source_intersecting_components,
            catchment_spanning_component_count: catchment_spanning_components,
            major_river_intersecting_component_count: river_intersecting_components,
            depression_intersecting_component_count: depression_intersecting_components,
            records: highland_records,
        },
        catchments: CatchmentSummaryV0 {
            policy: river_policy,
            effective_minimum_catchment_km2: river_policy.effective_all_minimum_km2(n),
            terminal_mouth_count: network.mouths.len(),
            support_assigned_cell_count: support_cells
                .iter()
                .filter(|&&cell| catchment_owner[cell].is_some())
                .count(),
            support_terminal_mouth_count: support_catchments.len(),
            boundary_proxy_cell_count: support_cells
                .iter()
                .filter(|&&cell| boundary_proxy[cell])
                .count(),
            boundary_proxy_area_km2: boundary_area,
            boundary_proxy_support_fraction: boundary_area / support_area.max(f64::MIN_POSITIVE),
        },
        trunks: TrunkSummaryV0 {
            selection_rule: "represented terminal-mouth main trunks ranked by support length, then mouth discharge, then lower mouth cell; first five retained in the sidecar",
            support_intersecting_trunk_count,
            recorded_trunk_count: recorded_trunks.len(),
            records: recorded_trunks,
        },
        depressions: DepressionSummaryV0 {
            support_intersecting_count: depressions.len(),
            source_intersecting_count: source_intersecting_depressions,
            highland_intersecting_count: highland_intersecting_depressions,
            records: depressions,
        },
        diagnostic_legend: DiagnosticLegendV0 {
            source_topology: "lineage 0 red, 1 green, linked lineage 2 blue; brightness is log source density; 1.5 km highland boundaries white; two-sweep endpoints magenta",
            catchment_divide_basin: "terminal-mouth catchment hues; graph boundary proxies white; major rivers cyan; topographic depression cells blue",
        },
    };
    Ok(RdsRelationshipAnalysisV0 {
        source_topology_colors: source_colors(
            source,
            support,
            &highland_mask,
            &endpoint_mask,
            tessellation,
        ),
        catchment_divide_basin_colors: catchment_colors(
            support,
            &catchment_owner,
            &boundary_proxy,
            &network.major_cells,
            &surface.hydrology.basin_id,
        ),
        summary,
    })
}

fn two_sweep_endpoints(tessellation: &Tessellation, cells: &[usize]) -> [usize; 2] {
    let start = *cells.iter().min().expect("component is nonempty");
    let farthest = |origin: usize| {
        let center = tessellation.cell_center(origin);
        cells
            .iter()
            .copied()
            .max_by(|&a, &b| {
                center
                    .dot(tessellation.cell_center(a))
                    .clamp(-1.0, 1.0)
                    .acos()
                    .total_cmp(
                        &center
                            .dot(tessellation.cell_center(b))
                            .clamp(-1.0, 1.0)
                            .acos(),
                    )
                    .then_with(|| b.cmp(&a))
            })
            .unwrap()
    };
    let a = farthest(start);
    [a, farthest(a)]
}

fn weighted_sum(cells: &[usize], area_km2: &[f64], value: impl Fn(usize) -> f64) -> f64 {
    cells.iter().map(|&cell| area_km2[cell] * value(cell)).sum()
}

fn dominant_lineage_for_cells(
    cells: &[usize],
    area_km2: &[f64],
    source: &RdsScheduleCompressionV0,
) -> (Option<u8>, f64) {
    let mut rates = [0.0; 3];
    for &cell in cells {
        if let Some(lineage) = source.dominant_lineage[cell] {
            rates[lineage as usize] += source.mean_source_density_per_myr[cell]
                * area_km2[cell]
                * f64::from(source.dominant_lineage_share[cell]);
        }
    }
    let total: f64 = rates.iter().sum();
    if total <= SOURCE_EPSILON {
        (None, 0.0)
    } else {
        let winner = (0..3)
            .max_by(|&a, &b| rates[a].total_cmp(&rates[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        (Some(winner as u8), rates[winner] / total)
    }
}

fn downstream_anchor_owner(
    drainage: &[Option<usize>],
    anchors: &[Option<usize>],
) -> Vec<Option<usize>> {
    let n = drainage.len();
    let mut owner = anchors.to_vec();
    let mut resolved: Vec<bool> = anchors.iter().map(Option::is_some).collect();
    let mut visit = vec![0u32; n];
    let mut generation = 0u32;
    for start in 0..n {
        if resolved[start] {
            continue;
        }
        generation = generation.wrapping_add(1);
        if generation == 0 {
            visit.fill(0);
            generation = 1;
        }
        let mut trail = Vec::new();
        let mut cursor = start;
        let terminal = loop {
            if resolved[cursor] {
                break owner[cursor];
            }
            if visit[cursor] == generation {
                break None;
            }
            visit[cursor] = generation;
            trail.push(cursor);
            match drainage[cursor] {
                Some(next) if next < n => cursor = next,
                _ => break None,
            }
        };
        for cell in trail {
            owner[cell] = terminal;
            resolved[cell] = true;
        }
    }
    owner
}

fn catchment_boundary_proxy(
    tessellation: &Tessellation,
    support: &[bool],
    owner: &[Option<usize>],
) -> Vec<bool> {
    let mut boundary = vec![false; support.len()];
    for cell in 0..support.len() {
        let Some(a) = support[cell].then_some(owner[cell]).flatten() else {
            continue;
        };
        for &neighbor in tessellation.neighbors(cell) {
            if support[neighbor] && owner[neighbor].is_some_and(|b| b != a) {
                boundary[cell] = true;
                boundary[neighbor] = true;
            }
        }
    }
    boundary
}

fn main_trunk(mouth: usize, hydrology: &Hydrology, network: &RiverNetwork) -> Vec<usize> {
    let mut trunk = vec![mouth];
    let mut current = mouth;
    while let Some(&upstream) = network.upstream[current].iter().max_by(|&&a, &&b| {
        hydrology.flow_accumulation[a]
            .total_cmp(&hydrology.flow_accumulation[b])
            .then_with(|| b.cmp(&a))
    }) {
        trunk.push(upstream);
        current = upstream;
        assert!(trunk.len() <= network.all_cells.len());
    }
    trunk.reverse();
    trunk
}

fn trunk_records(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    network: &RiverNetwork,
    support: &[bool],
    source: &RdsScheduleCompressionV0,
    area_km2: &[f64],
) -> Vec<TrunkRecordV0> {
    let mut records = Vec::new();
    for &mouth in &network.mouths {
        let trunk = main_trunk(mouth, hydrology, network);
        let support_cells: Vec<usize> = trunk.iter().copied().filter(|&c| support[c]).collect();
        if support_cells.is_empty() {
            continue;
        }
        let mut length = 0.0;
        let mut crossings = 0;
        let mut angle_sum = 0.0;
        let mut angle_count = 0;
        for edge in trunk.windows(2) {
            let [a, b] = [edge[0], edge[1]];
            crossings += usize::from(support[a] != support[b]);
            if !(support[a] && support[b]) {
                continue;
            }
            let pa = tessellation.cell_center(a);
            let pb = tessellation.cell_center(b);
            length += f64::from(pa.dot(pb).clamp(-1.0, 1.0).acos() * PLANET_RADIUS_KM);
            let midpoint = (pa + pb).normalize_or_zero();
            let segment = (pb - pa).reject_from(midpoint).normalize_or_zero();
            let fabric = (source.axial_fabric[a] + source.axial_fabric[b])
                .reject_from(midpoint)
                .normalize_or_zero();
            if segment.length_squared() > 0.0 && fabric.length_squared() > 0.0 {
                angle_sum += f64::from(
                    segment
                        .dot(fabric)
                        .abs()
                        .clamp(0.0, 1.0)
                        .acos()
                        .to_degrees(),
                );
                angle_count += 1;
            }
        }
        records.push(TrunkRecordV0 {
            mouth_cell: mouth,
            head_cell: trunk[0],
            strahler_order_at_mouth: network.strahler_order[mouth],
            mouth_discharge_equivalent_km2: hydrology.flow_accumulation[mouth]
                * PLANET_RADIUS_KM.powi(2),
            trunk_cell_count: trunk.len(),
            support_cell_count: support_cells.len(),
            support_length_km: length,
            support_boundary_crossings: crossings,
            source_rate_km2_per_myr: weighted_sum(&support_cells, area_km2, |cell| {
                source.mean_source_density_per_myr[cell]
            }),
            source_fabric_segment_count: angle_count,
            source_fabric_mean_acute_angle_deg: (angle_count > 0)
                .then_some(angle_sum / angle_count.max(1) as f64),
        });
    }
    records.sort_by(|a, b| {
        b.support_length_km
            .total_cmp(&a.support_length_km)
            .then_with(|| {
                b.mouth_discharge_equivalent_km2
                    .total_cmp(&a.mouth_discharge_equivalent_km2)
            })
            .then_with(|| a.mouth_cell.cmp(&b.mouth_cell))
    });
    records
}

fn depression_records(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    support: &[bool],
    highland: &[bool],
    source: &RdsScheduleCompressionV0,
    area_km2: &[f64],
) -> Vec<DepressionRecordV0> {
    let mut records = Vec::new();
    for (basin_id, basin) in hydrology.basins.iter().enumerate() {
        let cells: Vec<usize> = basin
            .cells
            .iter()
            .copied()
            .filter(|&c| support[c])
            .collect();
        if cells.is_empty() {
            continue;
        }
        let boundary: Vec<usize> = cells
            .iter()
            .copied()
            .filter(|&cell| {
                tessellation
                    .neighbors(cell)
                    .iter()
                    .any(|&n| hydrology.basin_id[n] != Some(basin_id))
            })
            .collect();
        let boundary_set: BTreeSet<usize> = boundary.iter().copied().collect();
        let interior: Vec<usize> = cells
            .iter()
            .copied()
            .filter(|cell| !boundary_set.contains(cell))
            .collect();
        let density = |set: &[usize]| {
            let area = weighted_sum(set, area_km2, |_| 1.0);
            weighted_sum(set, area_km2, |cell| {
                source.mean_source_density_per_myr[cell]
            }) / area.max(f64::MIN_POSITIVE)
        };
        let boundary_density = density(&boundary);
        let interior_density = density(&interior);
        records.push(DepressionRecordV0 {
            basin_id,
            support_cell_count: cells.len(),
            support_area_km2: weighted_sum(&cells, area_km2, |_| 1.0),
            highland_cell_count: cells.iter().filter(|&&cell| highland[cell]).count(),
            source_active_cell_count: cells
                .iter()
                .filter(|&&cell| source.active_frame_count[cell] > 0)
                .count(),
            source_rate_km2_per_myr: weighted_sum(&cells, area_km2, |cell| {
                source.mean_source_density_per_myr[cell]
            }),
            boundary_source_density_per_myr: boundary_density,
            interior_source_density_per_myr: interior_density,
            boundary_to_interior_source_ratio: (interior_density > SOURCE_EPSILON)
                .then_some(boundary_density / interior_density),
            bottom_elevation_km: elevation_to_km(basin.bottom_elevation),
            spill_elevation_km: elevation_to_km(basin.spill_elevation),
            sill_relief_km: elevation_to_km(basin.spill_elevation - basin.bottom_elevation),
            water_level_km: elevation_to_km(basin.water_level),
            has_water: basin.has_water(),
            overflowing: basin.is_overflowing(),
        });
    }
    records.sort_by(|a, b| {
        b.support_area_km2
            .total_cmp(&a.support_area_km2)
            .then_with(|| a.basin_id.cmp(&b.basin_id))
    });
    records
}

fn effective_count(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.filter(|value| *value > 0.0).collect();
    let total: f64 = values.iter().sum();
    if total <= SOURCE_EPSILON {
        return 0.0;
    }
    (-values
        .iter()
        .map(|value| {
            let p = value / total;
            p * p.ln()
        })
        .sum::<f64>())
    .exp()
}

fn source_colors(
    source: &RdsScheduleCompressionV0,
    support: &[bool],
    highland: &[bool],
    endpoints: &[bool],
    tessellation: &Tessellation,
) -> Vec<Vec3> {
    let mut positive: Vec<f64> = source
        .mean_source_density_per_myr
        .iter()
        .copied()
        .filter(|value| *value > SOURCE_EPSILON)
        .collect();
    positive.sort_unstable_by(f64::total_cmp);
    let scale = positive
        .get(((positive.len().saturating_sub(1)) as f64 * 0.95).round() as usize)
        .copied()
        .unwrap_or(1.0)
        .max(SOURCE_EPSILON);
    (0..support.len())
        .map(|cell| {
            if !support[cell] {
                return Vec3::splat(0.015);
            }
            let mut color = Vec3::splat(0.06);
            if let Some(lineage) = source.dominant_lineage[cell] {
                let intensity = ((1.0 + source.mean_source_density_per_myr[cell]).ln()
                    / (1.0 + scale).ln())
                .clamp(0.0, 1.0) as f32;
                let base = [
                    Vec3::new(0.95, 0.12, 0.08),
                    Vec3::new(0.10, 0.85, 0.18),
                    Vec3::new(0.12, 0.32, 1.0),
                ][lineage as usize];
                color = base * intensity * (0.45 + 0.55 * source.dominant_lineage_share[cell]);
            } else if source.mean_source_density_per_myr[cell] > SOURCE_EPSILON {
                let intensity = ((1.0 + source.mean_source_density_per_myr[cell]).ln()
                    / (1.0 + scale).ln())
                .clamp(0.0, 1.0) as f32;
                color = Vec3::new(0.90, 0.62, 0.12) * intensity;
            }
            let highland_boundary = highland[cell]
                && tessellation
                    .neighbors(cell)
                    .iter()
                    .any(|&neighbor| !highland[neighbor]);
            if highland_boundary {
                color = Vec3::splat(0.92);
            }
            if endpoints[cell] {
                color = Vec3::new(1.0, 0.02, 0.82);
            }
            color
        })
        .collect()
}

fn catchment_colors(
    support: &[bool],
    owner: &[Option<usize>],
    boundary: &[bool],
    major_river: &[bool],
    basin_id: &[Option<usize>],
) -> Vec<Vec3> {
    (0..support.len())
        .map(|cell| {
            if !support[cell] {
                return Vec3::splat(0.015);
            }
            let mut color = owner[cell].map(stable_color).unwrap_or(Vec3::splat(0.07));
            if basin_id[cell].is_some() {
                color = color * 0.3 + Vec3::new(0.03, 0.12, 0.60);
            }
            if major_river[cell] {
                color = Vec3::new(0.0, 0.95, 1.0);
            }
            if boundary[cell] {
                color = Vec3::ONE;
            }
            color
        })
        .collect()
}

fn stable_color(id: usize) -> Vec3 {
    let mut x = (id as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d_49bb_1331_11eb);
    x ^= x >> 31;
    let channel = |shift: u32| 0.20 + 0.60 * (((x >> shift) & 0xff_u64) as f32 / 255.0);
    Vec3::new(channel(0), channel(8), channel(16))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex3::world::{
        BoundaryEdgeId, RegionalDeformationCellContributionV0, RegionalDeformationElementIdV0,
        RegionalDeformationRasterLedgerV0,
    };

    fn frame(density: Vec<f64>, lineage: &[Option<(u8, f64)>]) -> RegionalDeformationRasterV0 {
        let n = density.len();
        RegionalDeformationRasterV0 {
            frame_index: Some(0),
            rate_density_per_myr: density,
            active_support_fraction: vec![0.0; n],
            axial_fabric: vec![Vec3::X; n],
            provenance: lineage
                .iter()
                .map(|entry| {
                    entry
                        .map(|(lineage, rate)| {
                            vec![RegionalDeformationCellContributionV0 {
                                element_id: RegionalDeformationElementIdV0 {
                                    parent_segment_id: BoundaryEdgeId::new(0, 1),
                                    lineage,
                                    side_ordinal: 0,
                                },
                                rate_density_per_myr: rate,
                            }]
                        })
                        .unwrap_or_default()
                })
                .collect(),
            ledger: RegionalDeformationRasterLedgerV0 {
                frame_index: Some(0),
                requested_flux_km2_per_myr: 0.0,
                allocated_flux_km2_per_myr: 0.0,
                unallocated_flux_km2_per_myr: 0.0,
                closure_residual_km2_per_myr: 0.0,
                active_cell_count: 0,
                additive_overlap_cell_count: 0,
            },
            omissions: Vec::new(),
        }
    }

    #[test]
    fn schedule_compression_uses_contributions_and_time_activity() {
        let a = frame(vec![2.0, 0.0], &[Some((0, 2.0)), None]);
        let b = frame(vec![4.0, 3.0], &[Some((2, 4.0)), Some((1, 3.0))]);
        let compressed = compress_rds0_schedule_v0(&[&a, &b]).unwrap();
        assert_eq!(compressed.mean_source_density_per_myr, vec![3.0, 1.5]);
        assert_eq!(compressed.active_frame_count, vec![2, 1]);
        assert_eq!(compressed.dominant_lineage, vec![Some(2), Some(1)]);
        assert!((compressed.dominant_lineage_share[0] - 2.0 / 3.0).abs() < 1.0e-6);
        assert_eq!(compressed.axial_fabric, vec![Vec3::X; 2]);
    }

    #[test]
    fn downstream_anchor_ownership_is_cycle_safe() {
        let drainage = vec![Some(1), Some(2), None, Some(4), Some(3), None];
        let anchors = vec![None, None, Some(2), None, None, None];
        assert_eq!(
            downstream_anchor_owner(&drainage, &anchors),
            vec![Some(2), Some(2), Some(2), None, None, None]
        );
    }
}
