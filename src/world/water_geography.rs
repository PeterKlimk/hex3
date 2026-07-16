//! Compact whole-world water-geography account derived from retained hydrology.
//!
//! This module adds no physical state or persistent cross-stage identity. It
//! summarizes one retained hydrology stage and gives its components and
//! relationships compact, deterministic identities so causal comparisons can
//! use the same definitions for oceans, landmasses, lakes, rivers, coastlines,
//! spill routes, and drainage-integration provenance.

use std::collections::VecDeque;

use serde::Serialize;

use super::{
    elevation_to_km, solid_angle_to_km2, Hydrology, RiverNetwork, SemanticWaterKind, Tessellation,
    WaterBodyId, WaterBodySemantics, WaterOutlet, PLANET_RADIUS_KM,
};

pub const WATER_GEOGRAPHY_REPORT_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize)]
pub struct WaterGeographyReport {
    pub schema_version: u32,
    pub cell_count: usize,
    pub ocean: OceanGeographySummary,
    pub land: LandGeographySummary,
    pub inland_water: InlandWaterSummary,
    pub basins: BasinSummary,
    pub rivers: RiverGeographySummary,
    pub integration: DrainageIntegrationSummary,
    pub consistency: WaterGeographyConsistency,
    /// Hydrologic ocean components, ordered by descending area then anchor cell.
    pub ocean_components: Vec<OceanComponentObject>,
    /// Geographic land components, ordered by descending area then anchor cell.
    /// This is a scale-neutral hierarchy, not a continent/island classification.
    pub landmasses: Vec<LandmassObject>,
    pub ocean_coasts: Vec<OceanCoastRelation>,
    pub basin_spills: Vec<BasinSpillRelation>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OceanComponentObject {
    pub water_body_id: WaterBodyId,
    pub anchor_cell: usize,
    pub cell_count: usize,
    pub area_km2: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct LandmassObject {
    pub anchor_cell: usize,
    pub cell_count: usize,
    pub area_km2: f32,
    pub ocean_coastline_km: f32,
    pub inland_shoreline_km: f32,
    pub adjacent_ocean_water_body_ids: Vec<WaterBodyId>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OceanCoastRelation {
    pub landmass_anchor_cell: usize,
    pub ocean_water_body_id: WaterBodyId,
    pub edge_count: usize,
    pub length_km: f32,
    pub anchor_land_cell: usize,
    pub anchor_ocean_cell: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct BasinSpillRelation {
    pub basin_id: usize,
    pub inland_water_body_ids: Vec<WaterBodyId>,
    pub wet: bool,
    pub overflowing: bool,
    /// The first cell outside the basin, not the spill saddle itself.
    pub spill_target_cell: usize,
    pub spill_elevation_km: f32,
    pub water_elevation_km: f32,
    pub destination: BasinSpillDestination,
    /// Potential topographic route from `spill_target_cell`, whether or not the
    /// basin's present water level activates overflow.
    pub route_cell_count: usize,
    pub route_integration_cut_cell_count: usize,
    pub route_maximum_integration_cut_depth_km: f32,
    /// This retained basin overlaps cells marked as a pre-integration breached source.
    /// It does not reconstruct identity of the pre-integration basin or event.
    pub overlaps_breached_source_cells: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind", content = "target")]
pub enum BasinSpillDestination {
    Ocean(WaterBodyId),
    Basin(usize),
    Unresolved(BasinSpillUnresolvedReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BasinSpillUnresolvedReason {
    NoDrainage,
    SelfBasin,
    Cycle,
    MissingOceanOwner,
}

#[derive(Clone, Debug, Serialize)]
pub struct OceanGeographySummary {
    pub component_count: usize,
    /// Descending component areas; compact component objects are reported separately.
    pub component_areas_km2: Vec<f32>,
    pub total_area_km2: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct LandGeographySummary {
    pub component_count: usize,
    /// Descending component areas; geographic land means any non-submerged cell.
    /// Compact landmass objects are reported separately.
    pub component_areas_km2: Vec<f32>,
    pub total_area_km2: f32,
    /// Shared Voronoi boundary between geographic land and connected ocean.
    pub ocean_coastline_km: f32,
    /// Shared Voronoi boundary between geographic land and classified lake water.
    pub inland_shoreline_km: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct InlandWaterSummary {
    pub lake_count: usize,
    pub pond_count: usize,
    pub terminal_lake_count: usize,
    pub overflowing_lake_count: usize,
    pub total_lake_area_km2: f32,
    pub total_pond_area_km2: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct BasinSummary {
    pub basin_count: usize,
    pub wet_basin_count: usize,
    pub dry_basin_count: usize,
    pub terminal_basin_count: usize,
    /// Full basins with no basin target. This normally means ocean, but the
    /// retained hydrology also uses `None` for an unresolved/cyclic walk.
    pub overflowing_without_basin_target_count: usize,
    pub overflowing_to_basin_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RiverGeographySummary {
    pub reach_count: usize,
    pub major_reach_count: usize,
    pub mouth_count: usize,
    pub ocean_mouth_count: usize,
    pub lake_mouth_count: usize,
    pub inland_mouth_count: usize,
    pub maximum_strahler_order: u8,
    /// These roles are selected independently and may name the same mouth.
    pub highest_discharge: Option<RiverRoleSummary>,
    pub longest_trunk: Option<RiverRoleSummary>,
    pub highest_order: Option<RiverRoleSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RiverRoleSummary {
    pub mouth_cell: usize,
    pub discharge_equivalent_km2: f32,
    pub strahler_order: u8,
    pub trunk_cell_count: usize,
    pub trunk_length_km_approx: f32,
    pub integration_cut_cell_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct DrainageIntegrationSummary {
    pub cut_cell_count: usize,
    pub cut_footprint_km2: f32,
    pub maximum_cut_depth_km: f32,
    pub breached_source_cell_count: usize,
    pub breached_source_footprint_km2: f32,
    pub river_channel_cut_cell_count: usize,
    pub major_river_cut_cell_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaterGeographyConsistency {
    /// Submerged cells with no semantic water-body owner.
    pub submerged_cells_without_owner: usize,
    /// Non-submerged cells unexpectedly assigned to an ocean or lake body.
    /// Pond ownership is intentional and counted separately.
    pub non_submerged_cells_with_wet_owner: usize,
    pub pond_owned_cell_count: usize,
    /// Body membership entries that disagree with `cell_body`.
    pub body_membership_mismatches: usize,
    /// Semantic bodies whose kind or members disagree with hydrologic state.
    pub body_state_mismatches: usize,
    /// Network mouths lacking a submerged downstream cell or semantic owner.
    pub unresolved_river_mouths: usize,
    pub semantic_ocean_component_count: usize,
    pub hydrologic_ocean_component_count: usize,
}

impl WaterGeographyReport {
    pub fn build(
        tessellation: &Tessellation,
        hydrology: &Hydrology,
        water: &WaterBodySemantics,
        rivers: &RiverNetwork,
    ) -> Result<Self, String> {
        validate_inputs(tessellation, hydrology, water, rivers)?;

        let n = tessellation.num_cells();
        let areas = tessellation.cell_areas_ref();
        let ocean_mask: Vec<bool> = (0..n).map(|cell| hydrology.is_ocean(cell)).collect();
        let land_mask: Vec<bool> = (0..n).map(|cell| !hydrology.is_submerged(cell)).collect();
        let ocean_catalog = component_catalog(tessellation, &ocean_mask, areas);
        let land_catalog = component_catalog(tessellation, &land_mask, areas);
        let ocean_areas: Vec<f32> = ocean_catalog
            .iter()
            .map(|component| component.area_km2)
            .collect();
        let land_areas: Vec<f32> = land_catalog
            .iter()
            .map(|component| component.area_km2)
            .collect();

        let ocean_components = ocean_component_objects(&ocean_catalog, water)?;
        let (landmasses, ocean_coasts) =
            landmass_objects(tessellation, hydrology, water, &land_catalog)?;
        let ocean_coastline_km = ocean_coasts.iter().map(|coast| coast.length_km).sum();
        let inland_shoreline_km = landmasses
            .iter()
            .map(|landmass| landmass.inland_shoreline_km)
            .sum();
        let basin_spills = basin_spill_relations(hydrology, water);

        let mut inland_water = InlandWaterSummary {
            lake_count: 0,
            pond_count: 0,
            terminal_lake_count: 0,
            overflowing_lake_count: 0,
            total_lake_area_km2: 0.0,
            total_pond_area_km2: 0.0,
        };
        for body in &water.bodies {
            match body.kind {
                SemanticWaterKind::Ocean => {}
                SemanticWaterKind::Lake => {
                    inland_water.lake_count += 1;
                    inland_water.total_lake_area_km2 += body.area_km2;
                    if body.outlet == WaterOutlet::Terminal {
                        inland_water.terminal_lake_count += 1;
                    } else {
                        inland_water.overflowing_lake_count += 1;
                    }
                }
                SemanticWaterKind::Pond => {
                    inland_water.pond_count += 1;
                    inland_water.total_pond_area_km2 += body.area_km2;
                }
            }
        }

        let mut basin_summary = BasinSummary {
            basin_count: hydrology.basins.len(),
            wet_basin_count: 0,
            dry_basin_count: 0,
            terminal_basin_count: 0,
            overflowing_without_basin_target_count: 0,
            overflowing_to_basin_count: 0,
        };
        for basin in &hydrology.basins {
            if basin.has_water() {
                basin_summary.wet_basin_count += 1;
            } else {
                basin_summary.dry_basin_count += 1;
            }
            if !basin.is_overflowing() {
                basin_summary.terminal_basin_count += 1;
            } else if basin.overflow_target.is_some() {
                basin_summary.overflowing_to_basin_count += 1;
            } else {
                basin_summary.overflowing_without_basin_target_count += 1;
            }
        }

        let discharge_candidates: Vec<RiverRoleSummary> = rivers
            .mouths
            .iter()
            .copied()
            .map(|mouth| {
                river_role(
                    tessellation,
                    hydrology,
                    rivers,
                    mouth,
                    TrunkCriterion::Discharge,
                    None,
                )
            })
            .collect();
        let longest_upstream = longest_upstream_links(tessellation, hydrology, rivers);
        let length_candidates: Vec<RiverRoleSummary> = rivers
            .mouths
            .iter()
            .copied()
            .map(|mouth| {
                river_role(
                    tessellation,
                    hydrology,
                    rivers,
                    mouth,
                    TrunkCriterion::Length,
                    Some(&longest_upstream),
                )
            })
            .collect();
        let order_candidates: Vec<RiverRoleSummary> = rivers
            .mouths
            .iter()
            .copied()
            .map(|mouth| {
                river_role(
                    tessellation,
                    hydrology,
                    rivers,
                    mouth,
                    TrunkCriterion::Order,
                    None,
                )
            })
            .collect();
        let highest_discharge = select_role(&discharge_candidates, |a, b| {
            a.discharge_equivalent_km2
                .total_cmp(&b.discharge_equivalent_km2)
                .then_with(|| a.strahler_order.cmp(&b.strahler_order))
                .then_with(|| b.mouth_cell.cmp(&a.mouth_cell))
        });
        let longest_trunk = select_role(&length_candidates, |a, b| {
            a.trunk_length_km_approx
                .total_cmp(&b.trunk_length_km_approx)
                .then_with(|| {
                    a.discharge_equivalent_km2
                        .total_cmp(&b.discharge_equivalent_km2)
                })
                .then_with(|| b.mouth_cell.cmp(&a.mouth_cell))
        });
        let highest_order = select_role(&order_candidates, |a, b| {
            a.strahler_order
                .cmp(&b.strahler_order)
                .then_with(|| {
                    a.discharge_equivalent_km2
                        .total_cmp(&b.discharge_equivalent_km2)
                })
                .then_with(|| b.mouth_cell.cmp(&a.mouth_cell))
        });

        let mut ocean_mouth_count = 0;
        let mut lake_mouth_count = 0;
        let mut inland_mouth_count = 0;
        let mut unresolved_river_mouths = 0;
        for &mouth in &rivers.mouths {
            match hydrology.downstream(mouth) {
                Some(next) if hydrology.is_ocean(next) => {
                    ocean_mouth_count += 1;
                    if water.cell_body[next].is_none() {
                        unresolved_river_mouths += 1;
                    }
                }
                Some(next) if hydrology.is_lake_water(next) => {
                    lake_mouth_count += 1;
                    if water.cell_body[next].is_none() {
                        unresolved_river_mouths += 1;
                    }
                }
                _ => {
                    inland_mouth_count += 1;
                    unresolved_river_mouths += 1;
                }
            }
        }

        let river_summary = RiverGeographySummary {
            reach_count: rivers.reaches.len(),
            major_reach_count: rivers.reaches.iter().filter(|reach| reach.is_major).count(),
            mouth_count: rivers.mouths.len(),
            ocean_mouth_count,
            lake_mouth_count,
            inland_mouth_count,
            maximum_strahler_order: rivers.strahler_order.iter().copied().max().unwrap_or(0),
            highest_discharge,
            longest_trunk,
            highest_order,
        };

        let cut_cell_count = hydrology.integration_cut_count();
        let integration = DrainageIntegrationSummary {
            cut_cell_count,
            cut_footprint_km2: hydrology
                .integration_cuts()
                .map(|(cell, _, _)| solid_angle_to_km2(areas[cell]))
                .sum(),
            maximum_cut_depth_km: hydrology
                .integration_cuts()
                .map(|(cell, _, _)| elevation_to_km(hydrology.integration_cut_depth(cell)))
                .fold(0.0, f32::max),
            breached_source_cell_count: hydrology
                .integration_breached_source
                .iter()
                .filter(|&&value| value)
                .count(),
            breached_source_footprint_km2: hydrology
                .integration_breached_source
                .iter()
                .enumerate()
                .filter(|(_, value)| **value)
                .map(|(cell, _)| solid_angle_to_km2(areas[cell]))
                .sum(),
            river_channel_cut_cell_count: hydrology
                .integration_cuts()
                .filter(|(cell, _, _)| rivers.all_cells[*cell])
                .count(),
            major_river_cut_cell_count: hydrology
                .integration_cuts()
                .filter(|(cell, _, _)| rivers.major_cells[*cell])
                .count(),
        };

        let consistency =
            consistency_summary(hydrology, water, ocean_areas.len(), unresolved_river_mouths);

        Ok(Self {
            schema_version: WATER_GEOGRAPHY_REPORT_SCHEMA_VERSION,
            cell_count: n,
            ocean: OceanGeographySummary {
                component_count: ocean_areas.len(),
                total_area_km2: ocean_areas.iter().sum(),
                component_areas_km2: ocean_areas,
            },
            land: LandGeographySummary {
                component_count: land_areas.len(),
                total_area_km2: land_areas.iter().sum(),
                component_areas_km2: land_areas,
                ocean_coastline_km,
                inland_shoreline_km,
            },
            inland_water,
            basins: basin_summary,
            rivers: river_summary,
            integration,
            consistency,
            ocean_components,
            landmasses,
            ocean_coasts,
            basin_spills,
        })
    }
}

fn validate_inputs(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    water: &WaterBodySemantics,
    rivers: &RiverNetwork,
) -> Result<(), String> {
    let n = tessellation.num_cells();
    let hydrology_lengths = [
        hydrology.elevation.len(),
        hydrology.filled_elevation.len(),
        hydrology.drainage_dir.len(),
        hydrology.flow_accumulation.len(),
        hydrology.is_ocean.len(),
        hydrology.basin_id.len(),
        hydrology.cell_water_body.len(),
        hydrology.integration_breached_source.len(),
    ];
    if hydrology_lengths.iter().any(|&length| length != n) {
        return Err("water geography requires cell-aligned hydrology fields".into());
    }
    if water.cell_body.len() != n {
        return Err("water geography requires cell-aligned water semantics".into());
    }
    if rivers.all_cells.len() != n
        || rivers.major_cells.len() != n
        || rivers.upstream.len() != n
        || rivers.strahler_order.len() != n
    {
        return Err("water geography requires cell-aligned river semantics".into());
    }
    if rivers.mouths.iter().any(|&cell| cell >= n)
        || hydrology
            .drainage_dir
            .iter()
            .flatten()
            .any(|&cell| cell >= n)
        || hydrology
            .basin_id
            .iter()
            .flatten()
            .any(|&basin| basin >= hydrology.basins.len())
        || hydrology
            .basins
            .iter()
            .any(|basin| basin.spill_target_cell >= n || basin.cells.iter().any(|&cell| cell >= n))
        || water
            .cell_body
            .iter()
            .flatten()
            .any(|&body| body >= water.bodies.len())
        || water
            .bodies
            .iter()
            .flat_map(|body| body.cells.iter())
            .any(|&cell| cell >= n)
        || water.bodies.iter().any(|body| {
            body.id
                .basin_id
                .is_some_and(|basin| basin >= hydrology.basins.len())
        })
    {
        return Err("water geography semantic object contains an out-of-range cell".into());
    }
    Ok(())
}

#[derive(Debug)]
struct ComponentCatalogEntry {
    members: Vec<usize>,
    anchor_cell: usize,
    area_km2: f32,
}

fn component_catalog(
    tessellation: &Tessellation,
    mask: &[bool],
    areas: &[f32],
) -> Vec<ComponentCatalogEntry> {
    let mut visited = vec![false; mask.len()];
    let mut result = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut area = 0.0;
        let mut members = Vec::new();
        while let Some(cell) = queue.pop_front() {
            area += solid_angle_to_km2(areas[cell]);
            members.push(cell);
            for &next in tessellation.neighbors(cell) {
                if mask[next] && !visited[next] {
                    visited[next] = true;
                    queue.push_back(next);
                }
            }
        }
        members.sort_unstable();
        result.push(ComponentCatalogEntry {
            anchor_cell: members[0],
            members,
            area_km2: area,
        });
    }
    result.sort_by(|a, b| {
        b.area_km2
            .total_cmp(&a.area_km2)
            .then_with(|| a.anchor_cell.cmp(&b.anchor_cell))
    });
    result
}

fn semantic_ocean_id(water: &WaterBodySemantics, cell: usize) -> Option<WaterBodyId> {
    let body = water.cell_body[cell].and_then(|index| water.bodies.get(index))?;
    (body.kind == SemanticWaterKind::Ocean).then_some(body.id)
}

fn ocean_component_objects(
    catalog: &[ComponentCatalogEntry],
    water: &WaterBodySemantics,
) -> Result<Vec<OceanComponentObject>, String> {
    catalog
        .iter()
        .map(|component| {
            let mut ids: Vec<WaterBodyId> = component
                .members
                .iter()
                .filter_map(|&cell| semantic_ocean_id(water, cell))
                .collect();
            sort_water_body_ids(&mut ids);
            ids.dedup();
            if ids.len() != 1 {
                return Err(format!(
                    "hydrologic ocean component at cell {} has {} semantic ocean owners",
                    component.anchor_cell,
                    ids.len()
                ));
            }
            Ok(OceanComponentObject {
                water_body_id: ids[0],
                anchor_cell: component.anchor_cell,
                cell_count: component.members.len(),
                area_km2: component.area_km2,
            })
        })
        .collect()
}

fn sort_water_body_ids(ids: &mut [WaterBodyId]) {
    ids.sort_by_key(|id| (id.basin_id, id.anchor_cell));
}

fn landmass_objects(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    water: &WaterBodySemantics,
    catalog: &[ComponentCatalogEntry],
) -> Result<(Vec<LandmassObject>, Vec<OceanCoastRelation>), String> {
    let mut landmasses = Vec::with_capacity(catalog.len());
    let mut all_relations = Vec::new();
    for component in catalog {
        let mut inland_shoreline_km = 0.0;
        let mut relations: Vec<OceanCoastRelation> = Vec::new();
        for &cell in &component.members {
            for &next in tessellation.neighbors(cell) {
                let length_km = tessellation.shared_edge_length(cell, next) * PLANET_RADIUS_KM;
                if hydrology.is_ocean(next) {
                    let ocean_id = semantic_ocean_id(water, next).ok_or_else(|| {
                        format!("ocean coast cell {next} lacks a semantic ocean owner")
                    })?;
                    if let Some(relation) = relations
                        .iter_mut()
                        .find(|relation| relation.ocean_water_body_id == ocean_id)
                    {
                        relation.edge_count += 1;
                        relation.length_km += length_km;
                        if (cell, next) < (relation.anchor_land_cell, relation.anchor_ocean_cell) {
                            relation.anchor_land_cell = cell;
                            relation.anchor_ocean_cell = next;
                        }
                    } else {
                        relations.push(OceanCoastRelation {
                            landmass_anchor_cell: component.anchor_cell,
                            ocean_water_body_id: ocean_id,
                            edge_count: 1,
                            length_km,
                            anchor_land_cell: cell,
                            anchor_ocean_cell: next,
                        });
                    }
                } else if hydrology.is_lake_water(next) {
                    inland_shoreline_km += length_km;
                }
            }
        }
        relations.sort_by_key(|relation| {
            (
                relation.ocean_water_body_id.basin_id,
                relation.ocean_water_body_id.anchor_cell,
            )
        });
        let ocean_coastline_km = relations.iter().map(|relation| relation.length_km).sum();
        let adjacent_ocean_water_body_ids = relations
            .iter()
            .map(|relation| relation.ocean_water_body_id)
            .collect();
        landmasses.push(LandmassObject {
            anchor_cell: component.anchor_cell,
            cell_count: component.members.len(),
            area_km2: component.area_km2,
            ocean_coastline_km,
            inland_shoreline_km,
            adjacent_ocean_water_body_ids,
        });
        all_relations.extend(relations);
    }
    Ok((landmasses, all_relations))
}

fn basin_spill_relations(
    hydrology: &Hydrology,
    water: &WaterBodySemantics,
) -> Vec<BasinSpillRelation> {
    let mut inland_ids = vec![Vec::new(); hydrology.basins.len()];
    for body in &water.bodies {
        if body.kind == SemanticWaterKind::Ocean {
            continue;
        }
        if let Some(basin_id) = body.id.basin_id {
            if let Some(ids) = inland_ids.get_mut(basin_id) {
                ids.push(body.id);
            }
        }
    }
    for ids in &mut inland_ids {
        sort_water_body_ids(ids);
        ids.dedup();
    }

    let ocean_owners: Vec<Option<WaterBodyId>> = (0..hydrology.elevation.len())
        .map(|cell| semantic_ocean_id(water, cell))
        .collect();
    let mut visit_stamps = vec![0usize; hydrology.elevation.len()];

    hydrology
        .basins
        .iter()
        .enumerate()
        .map(|(basin_id, basin)| {
            let trace = trace_spill_route(
                basin_id,
                basin.spill_target_cell,
                &hydrology.is_ocean,
                &hydrology.basin_id,
                &hydrology.drainage_dir,
                &ocean_owners,
                &mut visit_stamps,
                basin_id + 1,
            );
            let route_integration_cut_cell_count = trace
                .cells
                .iter()
                .filter(|&&cell| hydrology.was_lowered_by_integration(cell))
                .count();
            let route_maximum_integration_cut_depth_km = trace
                .cells
                .iter()
                .map(|&cell| elevation_to_km(hydrology.integration_cut_depth(cell)))
                .fold(0.0, f32::max);
            BasinSpillRelation {
                basin_id,
                inland_water_body_ids: inland_ids[basin_id].clone(),
                wet: basin.has_water(),
                overflowing: basin.is_overflowing(),
                spill_target_cell: basin.spill_target_cell,
                spill_elevation_km: elevation_to_km(basin.spill_elevation),
                water_elevation_km: elevation_to_km(basin.water_level),
                destination: trace.destination,
                route_cell_count: trace.cells.len(),
                route_integration_cut_cell_count,
                route_maximum_integration_cut_depth_km,
                overlaps_breached_source_cells: basin
                    .cells
                    .iter()
                    .any(|&cell| hydrology.integration_breached_source[cell]),
            }
        })
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
struct SpillTrace {
    destination: BasinSpillDestination,
    cells: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
fn trace_spill_route(
    source_basin: usize,
    target_cell: usize,
    is_ocean: &[bool],
    basin_id: &[Option<usize>],
    drainage_dir: &[Option<usize>],
    ocean_owners: &[Option<WaterBodyId>],
    visit_stamps: &mut [usize],
    stamp: usize,
) -> SpillTrace {
    let mut cells = Vec::new();
    let mut cell = target_cell;
    let destination = loop {
        if visit_stamps[cell] == stamp {
            break BasinSpillDestination::Unresolved(BasinSpillUnresolvedReason::Cycle);
        }
        visit_stamps[cell] = stamp;
        cells.push(cell);

        if is_ocean[cell] {
            break match ocean_owners[cell] {
                Some(id) => BasinSpillDestination::Ocean(id),
                None => {
                    BasinSpillDestination::Unresolved(BasinSpillUnresolvedReason::MissingOceanOwner)
                }
            };
        }
        if let Some(destination_basin) = basin_id[cell] {
            break if destination_basin == source_basin {
                BasinSpillDestination::Unresolved(BasinSpillUnresolvedReason::SelfBasin)
            } else {
                BasinSpillDestination::Basin(destination_basin)
            };
        }
        cell = match drainage_dir[cell] {
            Some(next) => next,
            None => {
                break BasinSpillDestination::Unresolved(BasinSpillUnresolvedReason::NoDrainage);
            }
        };
    };
    SpillTrace { destination, cells }
}

fn river_role(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    rivers: &RiverNetwork,
    mouth: usize,
    criterion: TrunkCriterion,
    longest_upstream: Option<&[Option<usize>]>,
) -> RiverRoleSummary {
    let mut trunk = vec![mouth];
    let mut current = mouth;
    while let Some(next) = match criterion {
        TrunkCriterion::Length => longest_upstream.expect("length links supplied")[current],
        TrunkCriterion::Discharge => rivers.upstream[current].iter().copied().max_by(|&a, &b| {
            hydrology.flow_accumulation[a]
                .total_cmp(&hydrology.flow_accumulation[b])
                .then_with(|| b.cmp(&a))
        }),
        TrunkCriterion::Order => rivers.upstream[current].iter().copied().max_by(|&a, &b| {
            rivers.strahler_order[a]
                .cmp(&rivers.strahler_order[b])
                .then_with(|| {
                    hydrology.flow_accumulation[a].total_cmp(&hydrology.flow_accumulation[b])
                })
                .then_with(|| b.cmp(&a))
        }),
    } {
        trunk.push(next);
        current = next;
    }
    trunk.reverse();
    RiverRoleSummary {
        mouth_cell: mouth,
        discharge_equivalent_km2: hydrology.flow_accumulation[mouth] * PLANET_RADIUS_KM.powi(2),
        strahler_order: rivers.strahler_order[mouth],
        trunk_cell_count: trunk.len(),
        trunk_length_km_approx: trunk
            .windows(2)
            .map(|pair| center_distance_km(tessellation, pair[0], pair[1]))
            .sum(),
        integration_cut_cell_count: trunk
            .iter()
            .filter(|&&cell| hydrology.was_lowered_by_integration(cell))
            .count(),
    }
}

#[derive(Clone, Copy)]
enum TrunkCriterion {
    Discharge,
    Length,
    Order,
}

fn longest_upstream_links(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    rivers: &RiverNetwork,
) -> Vec<Option<usize>> {
    let n = rivers.all_cells.len();
    let mut remaining_upstream: Vec<usize> = rivers.upstream.iter().map(Vec::len).collect();
    let mut queue: VecDeque<usize> = (0..n)
        .filter(|&cell| rivers.all_cells[cell] && remaining_upstream[cell] == 0)
        .collect();
    let mut distances = vec![0.0f32; n];
    let mut links = vec![None; n];
    while let Some(cell) = queue.pop_front() {
        let Some(downstream) = hydrology.downstream(cell) else {
            continue;
        };
        if !rivers.all_cells[downstream] {
            continue;
        }
        let candidate = distances[cell] + center_distance_km(tessellation, cell, downstream);
        let replace = match links[downstream] {
            None => true,
            Some(previous) => {
                candidate > distances[downstream]
                    || (candidate == distances[downstream] && cell < previous)
            }
        };
        if replace {
            distances[downstream] = candidate;
            links[downstream] = Some(cell);
        }
        remaining_upstream[downstream] -= 1;
        if remaining_upstream[downstream] == 0 {
            queue.push_back(downstream);
        }
    }
    links
}

fn center_distance_km(tessellation: &Tessellation, a: usize, b: usize) -> f32 {
    // Chord distance is stable on the fine mesh where f32 acos(dot) can round
    // short arcs to zero; the arc/chord difference is negligible at cell scale.
    (tessellation.cell_center(a) - tessellation.cell_center(b)).length() * PLANET_RADIUS_KM
}

fn select_role<F>(candidates: &[RiverRoleSummary], compare: F) -> Option<RiverRoleSummary>
where
    F: Fn(&RiverRoleSummary, &RiverRoleSummary) -> std::cmp::Ordering,
{
    candidates.iter().max_by(|a, b| compare(a, b)).cloned()
}

fn consistency_summary(
    hydrology: &Hydrology,
    water: &WaterBodySemantics,
    hydrologic_ocean_component_count: usize,
    unresolved_river_mouths: usize,
) -> WaterGeographyConsistency {
    let mut submerged_cells_without_owner = 0;
    let mut non_submerged_cells_with_wet_owner = 0;
    let mut pond_owned_cell_count = 0;
    for cell in 0..water.cell_body.len() {
        if hydrology.is_submerged(cell) && water.cell_body[cell].is_none() {
            submerged_cells_without_owner += 1;
        } else if !hydrology.is_submerged(cell) {
            if let Some(body_index) = water.cell_body[cell] {
                if water.bodies[body_index].kind == SemanticWaterKind::Pond {
                    pond_owned_cell_count += 1;
                } else {
                    non_submerged_cells_with_wet_owner += 1;
                }
            }
        }
    }

    let mut body_membership_mismatches = 0;
    let mut body_state_mismatches = 0;
    for (body_index, body) in water.bodies.iter().enumerate() {
        for &cell in &body.cells {
            if water.cell_body[cell] != Some(body_index) {
                body_membership_mismatches += 1;
            }
            let matches_state = match body.kind {
                SemanticWaterKind::Ocean => hydrology.is_ocean(cell),
                SemanticWaterKind::Lake => hydrology.is_lake_water(cell),
                SemanticWaterKind::Pond => !hydrology.is_submerged(cell),
            };
            if !matches_state {
                body_state_mismatches += 1;
            }
        }
    }

    WaterGeographyConsistency {
        submerged_cells_without_owner,
        non_submerged_cells_with_wet_owner,
        pond_owned_cell_count,
        body_membership_mismatches,
        body_state_mismatches,
        unresolved_river_mouths,
        semantic_ocean_component_count: water
            .bodies
            .iter()
            .filter(|body| body.kind == SemanticWaterKind::Ocean)
            .count(),
        hydrologic_ocean_component_count,
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::world::{Elevation, NoiseLayerData, RiverThresholdPolicy};

    #[test]
    fn generated_small_world_report_closes_its_counts_and_ownership() {
        let mut rng = ChaCha8Rng::seed_from_u64(73);
        let tessellation = Tessellation::generate(400, 0, &mut rng);
        let n = tessellation.num_cells();
        // A smooth north-land/south-ocean world avoids fixture-only assumptions
        // while exercising the complete retained hydrology and semantic path.
        let elevation = Elevation {
            values: (0..n)
                .map(|cell| 0.2 * tessellation.cell_center(cell).y)
                .collect(),
            noise_contribution: vec![0.0; n],
            noise_layers: NoiseLayerData {
                macro_layer: vec![0.0; n],
            },
        };
        let hydrology = Hydrology::generate_from_continentality(
            &tessellation,
            &vec![0.0; n],
            &elevation,
            &vec![1.0; n],
            &vec![0.5; n],
        );
        let water = WaterBodySemantics::build(&tessellation, &hydrology);
        let rivers = RiverNetwork::build(
            &tessellation,
            &hydrology,
            &water,
            RiverThresholdPolicy::default(),
        );
        let report =
            WaterGeographyReport::build(&tessellation, &hydrology, &water, &rivers).unwrap();

        assert_eq!(report.schema_version, 2);
        assert!(report.ocean.component_count > 0);
        assert!(report.land.component_count > 0);
        assert!(report.land.ocean_coastline_km > 0.0);
        assert_eq!(
            report.basins.basin_count,
            report.basins.wet_basin_count + report.basins.dry_basin_count
        );
        assert_eq!(
            report.basins.basin_count,
            report.basins.terminal_basin_count
                + report.basins.overflowing_to_basin_count
                + report.basins.overflowing_without_basin_target_count
        );
        assert_eq!(
            report.rivers.mouth_count,
            report.rivers.ocean_mouth_count
                + report.rivers.lake_mouth_count
                + report.rivers.inland_mouth_count
        );
        assert_eq!(
            report.integration.cut_cell_count,
            hydrology.integration_cut_count()
        );
        assert_eq!(report.consistency.submerged_cells_without_owner, 0);
        assert_eq!(report.consistency.non_submerged_cells_with_wet_owner, 0);
        assert_eq!(report.consistency.body_membership_mismatches, 0);
        assert_eq!(report.consistency.body_state_mismatches, 0);
        assert_eq!(
            report.consistency.semantic_ocean_component_count,
            report.consistency.hydrologic_ocean_component_count
        );

        assert_eq!(report.ocean_components.len(), report.ocean.component_count);
        assert_eq!(report.landmasses.len(), report.land.component_count);
        assert_eq!(
            report
                .ocean_components
                .iter()
                .map(|component| component.cell_count)
                .sum::<usize>(),
            (0..n).filter(|&cell| hydrology.is_ocean(cell)).count()
        );
        assert_eq!(
            report
                .landmasses
                .iter()
                .map(|landmass| landmass.cell_count)
                .sum::<usize>(),
            (0..n).filter(|&cell| !hydrology.is_submerged(cell)).count()
        );
        assert_close(
            report
                .ocean_components
                .iter()
                .map(|component| component.area_km2)
                .sum(),
            report.ocean.total_area_km2,
        );
        assert_close(
            report
                .landmasses
                .iter()
                .map(|landmass| landmass.area_km2)
                .sum(),
            report.land.total_area_km2,
        );

        let coast_length: f32 = report
            .ocean_coasts
            .iter()
            .map(|relation| relation.length_km)
            .sum();
        assert_eq!(coast_length, report.land.ocean_coastline_km);
        for relation in &report.ocean_coasts {
            assert!(relation.edge_count > 0);
            assert!(relation.length_km > 0.0);
            assert!(report
                .landmasses
                .iter()
                .any(|landmass| landmass.anchor_cell == relation.landmass_anchor_cell));
            assert!(water.bodies.iter().any(|body| {
                body.kind == SemanticWaterKind::Ocean && body.id == relation.ocean_water_body_id
            }));
        }

        assert_eq!(report.basin_spills.len(), hydrology.basins.len());
        for (basin_id, relation) in report.basin_spills.iter().enumerate() {
            let basin = &hydrology.basins[basin_id];
            assert_eq!(relation.basin_id, basin_id);
            assert_eq!(relation.spill_target_cell, basin.spill_target_cell);
            assert_eq!(relation.wet, basin.has_water());
            assert_eq!(relation.overflowing, basin.is_overflowing());
            assert!(relation.route_cell_count > 0);
            assert!(relation.route_integration_cut_cell_count <= relation.route_cell_count);
            assert!(relation.route_maximum_integration_cut_depth_km >= 0.0);
            assert!(
                relation.route_maximum_integration_cut_depth_km
                    <= report.integration.maximum_cut_depth_km
            );
            assert!(relation.inland_water_body_ids.iter().all(|id| {
                id.basin_id == Some(basin_id) && water.bodies.iter().any(|body| body.id == *id)
            }));
        }
        let unresolved: Vec<_> = report
            .basin_spills
            .iter()
            .filter_map(|spill| match spill.destination {
                BasinSpillDestination::Unresolved(reason) => Some((spill.basin_id, reason)),
                _ => None,
            })
            .collect();
        eprintln!("generated water-geography unresolved spills: {unresolved:?}");

        let second =
            WaterGeographyReport::build(&tessellation, &hydrology, &water, &rivers).unwrap();
        assert_eq!(
            serde_json::to_value(&report).unwrap(),
            serde_json::to_value(&second).unwrap()
        );
    }

    fn assert_close(actual: f32, expected: f32) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!((actual - expected).abs() <= 1.0e-6 * scale);
    }

    #[test]
    fn spill_trace_distinguishes_every_terminal_state() {
        let ocean_id = WaterBodyId {
            basin_id: None,
            anchor_cell: 3,
        };
        let is_ocean = vec![false, false, false, true, false, true];
        let basin_id = vec![None, None, Some(7), None, None, None];
        let ocean_owners = vec![None, None, None, Some(ocean_id), None, None];
        let mut stamps = vec![0; 6];

        let ocean = trace_spill_route(
            5,
            0,
            &is_ocean,
            &basin_id,
            &[Some(1), Some(3), None, None, None, None],
            &ocean_owners,
            &mut stamps,
            1,
        );
        assert_eq!(ocean.destination, BasinSpillDestination::Ocean(ocean_id));
        assert_eq!(ocean.cells, vec![0, 1, 3]);

        let basin = trace_spill_route(
            5,
            2,
            &is_ocean,
            &basin_id,
            &[None; 6],
            &ocean_owners,
            &mut stamps,
            2,
        );
        assert_eq!(basin.destination, BasinSpillDestination::Basin(7));

        let self_basin = trace_spill_route(
            7,
            2,
            &is_ocean,
            &basin_id,
            &[None; 6],
            &ocean_owners,
            &mut stamps,
            3,
        );
        assert_eq!(
            self_basin.destination,
            BasinSpillDestination::Unresolved(BasinSpillUnresolvedReason::SelfBasin)
        );

        let no_drainage = trace_spill_route(
            5,
            4,
            &is_ocean,
            &basin_id,
            &[None; 6],
            &ocean_owners,
            &mut stamps,
            4,
        );
        assert_eq!(
            no_drainage.destination,
            BasinSpillDestination::Unresolved(BasinSpillUnresolvedReason::NoDrainage)
        );

        let cycle = trace_spill_route(
            5,
            0,
            &is_ocean,
            &basin_id,
            &[Some(1), Some(0), None, None, None, None],
            &ocean_owners,
            &mut stamps,
            5,
        );
        assert_eq!(
            cycle.destination,
            BasinSpillDestination::Unresolved(BasinSpillUnresolvedReason::Cycle)
        );
        assert_eq!(cycle.cells, vec![0, 1]);

        let missing_owner = trace_spill_route(
            5,
            5,
            &is_ocean,
            &basin_id,
            &[None; 6],
            &ocean_owners,
            &mut stamps,
            6,
        );
        assert_eq!(
            missing_owner.destination,
            BasinSpillDestination::Unresolved(BasinSpillUnresolvedReason::MissingOceanOwner)
        );
    }
}
