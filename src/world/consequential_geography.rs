//! On-demand physical/access substrate for Consequential Geography V0.
//!
//! This module deliberately stops before site scoring or route-network
//! generation. It exposes raw, inspectable components so later authored priors
//! cannot hide compensation among terrain, water, coast and living opportunity.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::Serialize;

use super::{
    elevation_to_km, Hydrology, LivingSurfaceSemantics, RiverSelection, RiverThresholdPolicy,
    SemanticWaterKind, Tessellation, WaterBodySemantics, PLANET_RADIUS_KM,
};

#[derive(Clone, Copy, Debug, Serialize)]
pub struct TraversalConfig {
    /// Generalized-km cost per kilometre climbed.
    uphill_penalty: f32,
    /// Generalized-km cost per kilometre descended.
    downhill_penalty: f32,
}

impl TraversalConfig {
    pub fn new(uphill_penalty: f32, downhill_penalty: f32) -> Result<Self, &'static str> {
        if !uphill_penalty.is_finite()
            || !downhill_penalty.is_finite()
            || downhill_penalty < 0.0
            || uphill_penalty < downhill_penalty
        {
            return Err("traversal penalties must be finite with uphill >= downhill >= 0");
        }
        Ok(Self {
            uphill_penalty,
            downhill_penalty,
        })
    }

    pub fn uphill_penalty(self) -> f32 {
        self.uphill_penalty
    }

    pub fn downhill_penalty(self) -> f32 {
        self.downhill_penalty
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DirectedEdgeCost {
    pub from: usize,
    pub to: usize,
    pub distance_km: f32,
    pub elevation_change_km: f32,
    pub ascent_km: f32,
    pub descent_km: f32,
    pub signed_grade: f32,
    pub generalized_cost_km: f32,
    pub touches_drainage_repair: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConsequentialGeographyComponents {
    /// Land-only, direction-neutral generalized access burden to a selected
    /// river or proper-lake shore. `None` means no source is reachable.
    pub freshwater_access_generalized_km: Vec<Option<f32>>,
    /// Land-only, direction-neutral generalized access burden to ocean coast.
    pub coast_access_generalized_km: Vec<Option<f32>>,
    /// Exact source masks used by the two access fields.
    pub freshwater_source: Vec<bool>,
    pub coast_source: Vec<bool>,
    /// River-selection provenance for the freshwater source mask.
    pub aggregate_river_policy: RiverThresholdPolicy,
    /// Exact accepted Living Surface vegetation-cover opportunity, not yield or
    /// carrying capacity.
    pub relative_living_opportunity: Vec<f32>,
    pub drainage_saturation: Vec<f32>,
    pub relative_water_limitation: Vec<f32>,
    /// Provenance only: these cells use drainage-integrated effective terrain.
    pub drainage_repaired: Vec<bool>,
    pub traversal: TraversalConfig,
}

impl ConsequentialGeographyComponents {
    pub fn build(
        tessellation: &Tessellation,
        hydrology: &Hydrology,
        water: &WaterBodySemantics,
        rivers: &RiverSelection,
        living: &LivingSurfaceSemantics,
        traversal: TraversalConfig,
    ) -> Result<Self, &'static str> {
        let n = tessellation.num_cells();
        if hydrology.elevation.len() != n
            || hydrology.is_ocean.len() != n
            || hydrology.basin_id.len() != n
            || hydrology.cell_water_body.len() != n
            || water.cell_body.len() != n
            || rivers.all_cells.len() != n
            || living.cells.len() != n
        {
            return Err("consequential-geography input lengths must match tessellation");
        }
        if hydrology.elevation.iter().any(|value| !value.is_finite()) {
            return Err("consequential-geography terrain must be finite");
        }
        if hydrology
            .cell_water_body
            .iter()
            .flatten()
            .any(|&body| body >= hydrology.water_bodies.len())
        {
            return Err("consequential-geography hydrology water-body index is out of range");
        }
        if water
            .cell_body
            .iter()
            .flatten()
            .any(|&body| body >= water.bodies.len())
        {
            return Err("consequential-geography water-body index is out of range");
        }
        if living.cells.iter().any(|cell| {
            [
                cell.vegetation_cover,
                cell.drainage_saturation,
                cell.relative_water_limitation,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        }) {
            return Err("consequential-geography living components must be finite fractions");
        }

        let submerged: Vec<bool> = (0..n).map(|cell| hydrology.is_submerged(cell)).collect();
        let (freshwater_sources, coast_sources) =
            source_masks(tessellation, &submerged, water, &rivers.all_cells);

        let freshwater_access_generalized_km = access_costs(
            tessellation,
            &hydrology.elevation,
            &submerged,
            &freshwater_sources,
            traversal,
        )?;
        let coast_access_generalized_km = access_costs(
            tessellation,
            &hydrology.elevation,
            &submerged,
            &coast_sources,
            traversal,
        )?;

        Ok(Self {
            freshwater_access_generalized_km,
            coast_access_generalized_km,
            freshwater_source: freshwater_sources,
            coast_source: coast_sources,
            aggregate_river_policy: rivers.policy,
            relative_living_opportunity: living
                .cells
                .iter()
                .map(|cell| cell.vegetation_cover)
                .collect(),
            drainage_saturation: living
                .cells
                .iter()
                .map(|cell| cell.drainage_saturation)
                .collect(),
            relative_water_limitation: living
                .cells
                .iter()
                .map(|cell| cell.relative_water_limitation)
                .collect(),
            drainage_repaired: (0..n)
                .map(|cell| hydrology.was_lowered_by_integration(cell))
                .collect(),
            traversal,
        })
    }
}

fn source_masks(
    tessellation: &Tessellation,
    submerged: &[bool],
    water: &WaterBodySemantics,
    selected_rivers: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let n = tessellation.num_cells();
    let mut freshwater = vec![false; n];
    let mut coast = vec![false; n];
    for cell in 0..n {
        if submerged[cell] {
            continue;
        }
        freshwater[cell] = selected_rivers[cell];
        for &neighbor in tessellation.neighbors(cell) {
            let Some(body_index) = water.cell_body[neighbor] else {
                continue;
            };
            match water.bodies[body_index].kind {
                SemanticWaterKind::Ocean => coast[cell] = true,
                SemanticWaterKind::Lake => freshwater[cell] = true,
                SemanticWaterKind::Pond => {}
            }
        }
    }
    (freshwater, coast)
}

pub fn directed_edge_cost(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    from: usize,
    to: usize,
    config: TraversalConfig,
) -> Result<DirectedEdgeCost, &'static str> {
    if from >= tessellation.num_cells()
        || to >= tessellation.num_cells()
        || hydrology.elevation.len() != tessellation.num_cells()
        || !tessellation.neighbors(from).contains(&to)
    {
        return Err("directed traversal requires an in-range adjacent cell pair");
    }
    edge_cost_from_elevation(
        tessellation,
        &hydrology.elevation,
        from,
        to,
        hydrology.was_lowered_by_integration(from) || hydrology.was_lowered_by_integration(to),
        config,
    )
}

fn edge_cost_from_elevation(
    tessellation: &Tessellation,
    elevation: &[f32],
    from: usize,
    to: usize,
    touches_drainage_repair: bool,
    config: TraversalConfig,
) -> Result<DirectedEdgeCost, &'static str> {
    let chord = (tessellation.cell_center(from) - tessellation.cell_center(to)).length();
    let distance_km = 2.0 * PLANET_RADIUS_KM * (0.5 * chord).clamp(0.0, 1.0).asin();
    if !distance_km.is_finite() || distance_km <= 0.0 {
        return Err("adjacent traversal edge must have positive finite length");
    }
    let elevation_change_km = elevation_to_km(elevation[to] - elevation[from]);
    if !elevation_change_km.is_finite() {
        return Err("traversal elevation change must be finite");
    }
    let ascent_km = elevation_change_km.max(0.0);
    let descent_km = (-elevation_change_km).max(0.0);
    let generalized_cost_km =
        distance_km + config.uphill_penalty * ascent_km + config.downhill_penalty * descent_km;
    if !generalized_cost_km.is_finite() || generalized_cost_km < distance_km {
        return Err("generalized traversal cost must be finite and at least physical distance");
    }
    Ok(DirectedEdgeCost {
        from,
        to,
        distance_km,
        elevation_change_km,
        ascent_km,
        descent_km,
        signed_grade: elevation_change_km / distance_km,
        generalized_cost_km,
        touches_drainage_repair,
    })
}

#[derive(Clone, Copy, Debug)]
struct QueueEntry {
    cost: f32,
    cell: usize,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits() && self.cell == other.cell
    }
}
impl Eq for QueueEntry {}
impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

fn access_costs(
    tessellation: &Tessellation,
    elevation: &[f32],
    submerged: &[bool],
    sources: &[bool],
    config: TraversalConfig,
) -> Result<Vec<Option<f32>>, &'static str> {
    let n = tessellation.num_cells();
    if elevation.len() != n || submerged.len() != n || sources.len() != n {
        return Err("access-field inputs must match tessellation");
    }
    let mut distances = vec![f32::INFINITY; n];
    let mut queue = BinaryHeap::new();
    for cell in 0..n {
        if sources[cell] && !submerged[cell] {
            distances[cell] = 0.0;
            queue.push(QueueEntry { cost: 0.0, cell });
        }
    }
    while let Some(QueueEntry { cost, cell }) = queue.pop() {
        if cost > distances[cell] {
            continue;
        }
        for &neighbor in tessellation.neighbors(cell) {
            if submerged[neighbor] {
                continue;
            }
            let forward =
                edge_cost_from_elevation(tessellation, elevation, cell, neighbor, false, config)?;
            let reverse =
                edge_cost_from_elevation(tessellation, elevation, neighbor, cell, false, config)?;
            let next = cost + 0.5 * (forward.generalized_cost_km + reverse.generalized_cost_km);
            if next < distances[neighbor] {
                distances[neighbor] = next;
                queue.push(QueueEntry {
                    cost: next,
                    cell: neighbor,
                });
            }
        }
    }
    Ok(distances
        .into_iter()
        .map(|distance| distance.is_finite().then_some(distance))
        .collect())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::world::{SemanticWaterBody, WaterBodyId, WaterOutlet};

    fn tessellation(cells: usize) -> Tessellation {
        let mut rng = ChaCha8Rng::seed_from_u64(61_207);
        Tessellation::generate(cells, 0, &mut rng)
    }

    fn config() -> TraversalConfig {
        TraversalConfig::new(12.0, 3.0).unwrap()
    }

    fn one_cell_water(
        cell_count: usize,
        cell: usize,
        kind: SemanticWaterKind,
    ) -> WaterBodySemantics {
        let mut cell_body = vec![None; cell_count];
        cell_body[cell] = Some(0);
        WaterBodySemantics {
            bodies: vec![SemanticWaterBody {
                id: WaterBodyId {
                    basin_id: None,
                    anchor_cell: cell,
                },
                kind,
                cells: vec![cell],
                area_km2: 1.0,
                surface_elevation_km: 0.0,
                max_depth_km: 1.0,
                outlet: WaterOutlet::Terminal,
            }],
            cell_body,
        }
    }

    #[test]
    fn traversal_config_rejects_invalid_penalty_order_and_values() {
        assert!(TraversalConfig::new(2.0, 3.0).is_err());
        assert!(TraversalConfig::new(2.0, -1.0).is_err());
        assert!(TraversalConfig::new(f32::NAN, 0.0).is_err());
        let valid = TraversalConfig::new(3.0, 2.0).unwrap();
        assert_eq!(valid.uphill_penalty(), 3.0);
        assert_eq!(valid.downhill_penalty(), 2.0);
    }

    #[test]
    fn freshwater_and_coast_sources_keep_distinct_semantics() {
        let tess = tessellation(80);
        let n = tess.num_cells();
        let land = 0;
        let water_cell = tess.neighbors(land)[0];
        let mut submerged = vec![false; n];
        submerged[water_cell] = true;
        let no_rivers = vec![false; n];

        let ocean = one_cell_water(n, water_cell, SemanticWaterKind::Ocean);
        let (freshwater, coast) = source_masks(&tess, &submerged, &ocean, &no_rivers);
        assert!(!freshwater[land]);
        assert!(coast[land]);

        let lake = one_cell_water(n, water_cell, SemanticWaterKind::Lake);
        let (freshwater, coast) = source_masks(&tess, &submerged, &lake, &no_rivers);
        assert!(freshwater[land]);
        assert!(!coast[land]);

        let pond = one_cell_water(n, water_cell, SemanticWaterKind::Pond);
        let (freshwater, coast) = source_masks(&tess, &submerged, &pond, &no_rivers);
        assert!(!freshwater[land]);
        assert!(!coast[land]);

        let empty_water = WaterBodySemantics {
            bodies: Vec::new(),
            cell_body: vec![None; n],
        };
        let mut river = no_rivers;
        river[land] = true;
        river[water_cell] = true;
        let (freshwater, _) = source_masks(&tess, &submerged, &empty_water, &river);
        assert!(freshwater[land]);
        assert!(!freshwater[water_cell]);
    }

    #[test]
    fn flat_edges_cost_distance_and_reverse_slope_components_swap() {
        let tess = tessellation(80);
        let a = 0;
        let b = tess.neighbors(a)[0];
        let mut elevation = vec![0.0; tess.num_cells()];
        let flat = edge_cost_from_elevation(&tess, &elevation, a, b, false, config()).unwrap();
        assert_eq!(flat.generalized_cost_km, flat.distance_km);

        elevation[b] = 0.2;
        let up = edge_cost_from_elevation(&tess, &elevation, a, b, false, config()).unwrap();
        let down = edge_cost_from_elevation(&tess, &elevation, b, a, false, config()).unwrap();
        assert_eq!(up.distance_km.to_bits(), down.distance_km.to_bits());
        assert_eq!(up.ascent_km.to_bits(), down.descent_km.to_bits());
        assert_eq!(up.descent_km.to_bits(), down.ascent_km.to_bits());
        assert!(up.generalized_cost_km > down.generalized_cost_km);
    }

    #[test]
    fn neutral_access_is_physical_graph_distance_and_water_is_not_a_shortcut() {
        let tess = tessellation(100);
        let n = tess.num_cells();
        let elevation = vec![0.0; n];
        let mut sources = vec![false; n];
        sources[0] = true;
        let dry = vec![false; n];
        let access = access_costs(&tess, &elevation, &dry, &sources, config()).unwrap();
        assert_eq!(access[0], Some(0.0));
        assert!(access.iter().all(Option::is_some));

        let mut submerged = vec![true; n];
        submerged[0] = false;
        let isolated = access_costs(&tess, &elevation, &submerged, &sources, config()).unwrap();
        assert_eq!(isolated[0], Some(0.0));
        assert!(isolated.iter().skip(1).all(Option::is_none));
    }

    #[test]
    fn lower_gap_reduces_access_cost_across_a_steep_global_barrier() {
        let tess = tessellation(900);
        let n = tess.num_cells();
        let mut source = 0;
        let mut target = 0;
        for cell in 1..n {
            if tess.cell_center(cell).x < tess.cell_center(source).x {
                source = cell;
            }
            if tess.cell_center(cell).x > tess.cell_center(target).x {
                target = cell;
            }
        }
        let mut sources = vec![false; n];
        sources[source] = true;
        let submerged = vec![false; n];
        let mut closed = vec![0.0; n];
        for (cell, value) in closed.iter_mut().enumerate() {
            if tess.cell_center(cell).x.abs() < 0.28 {
                *value = 100.0;
            }
        }
        let mut open = closed.clone();
        for (cell, value) in open.iter_mut().enumerate() {
            let center = tess.cell_center(cell);
            if center.x.abs() < 0.28 && center.z > 0.72 {
                *value = 0.0;
            }
        }
        let closed_cost = access_costs(&tess, &closed, &submerged, &sources, config()).unwrap();
        let open_cost = access_costs(&tess, &open, &submerged, &sources, config()).unwrap();
        assert!(open_cost[target].unwrap() < closed_cost[target].unwrap());
    }
}
