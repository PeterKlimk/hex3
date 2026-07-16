//! Compact whole-world water-geography account derived from retained hydrology.
//!
//! This module adds no physical state or geographic identity. It summarizes one
//! tessellation and its already-derived water and river semantics so causal
//! comparisons can use the same definitions for oceans, land, lakes, rivers,
//! coastlines, and drainage-integration provenance.

use std::collections::VecDeque;

use serde::Serialize;

use super::{
    elevation_to_km, solid_angle_to_km2, Hydrology, RiverNetwork, SemanticWaterKind, Tessellation,
    WaterBodySemantics, WaterOutlet, PLANET_RADIUS_KM,
};

pub const WATER_GEOGRAPHY_REPORT_SCHEMA_VERSION: u32 = 1;

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
}

#[derive(Clone, Debug, Serialize)]
pub struct OceanGeographySummary {
    pub component_count: usize,
    /// Descending component areas; no per-cell membership is retained.
    pub component_areas_km2: Vec<f32>,
    pub total_area_km2: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct LandGeographySummary {
    pub component_count: usize,
    /// Descending component areas; geographic land means any non-submerged cell.
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
        let ocean_areas = component_areas(tessellation, &ocean_mask, areas);
        let land_areas = component_areas(tessellation, &land_mask, areas);

        let (ocean_coastline_km, inland_shoreline_km) =
            shoreline_lengths(tessellation, hydrology, &land_mask);

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
    {
        return Err("water geography semantic object contains an out-of-range cell".into());
    }
    Ok(())
}

fn component_areas(tessellation: &Tessellation, mask: &[bool], areas: &[f32]) -> Vec<f32> {
    let mut visited = vec![false; mask.len()];
    let mut result = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut area = 0.0;
        while let Some(cell) = queue.pop_front() {
            area += solid_angle_to_km2(areas[cell]);
            for &next in tessellation.neighbors(cell) {
                if mask[next] && !visited[next] {
                    visited[next] = true;
                    queue.push_back(next);
                }
            }
        }
        result.push(area);
    }
    result.sort_by(|a, b| b.total_cmp(a));
    result
}

fn shoreline_lengths(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    land_mask: &[bool],
) -> (f32, f32) {
    let mut ocean = 0.0;
    let mut inland = 0.0;
    for (cell, &is_land) in land_mask.iter().enumerate() {
        if !is_land {
            continue;
        }
        for &next in tessellation.neighbors(cell) {
            if hydrology.is_ocean(next) {
                ocean += tessellation.shared_edge_length(cell, next) * PLANET_RADIUS_KM;
            } else if hydrology.is_lake_water(next) {
                inland += tessellation.shared_edge_length(cell, next) * PLANET_RADIUS_KM;
            }
        }
    }
    (ocean, inland)
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
    }
}
