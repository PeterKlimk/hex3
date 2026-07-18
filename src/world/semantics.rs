//! Reusable semantic objects derived from modeled world state.
//!
//! These types interpret hydrology without changing it. They are independent of
//! camera, line width, color, relief scale, and other presentation settings.

use std::collections::VecDeque;

use serde::Serialize;

use super::{elevation_to_km, solid_angle_to_km2, Hydrology, Tessellation, PLANET_RADIUS_KM};

pub const DEFAULT_RIVER_MIN_CATCHMENT_KM2: f32 = 2_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum RiverThresholdPolicy {
    /// Historical count-equivalent thresholds, retained for controlled A/Bs.
    Legacy {
        all_fraction: f32,
        outlet_fraction: f32,
        branch_fraction: f32,
    },
    /// Physical catchment thresholds at land-mean wetness. The effective
    /// minimum is clamped to four global-mean cells when the mesh is too coarse
    /// to represent the requested scale.
    CatchmentKm2 {
        minimum: f32,
        major_outlet_multiplier: f32,
        major_branch_multiplier: f32,
    },
}

impl Default for RiverThresholdPolicy {
    fn default() -> Self {
        Self::catchment(DEFAULT_RIVER_MIN_CATCHMENT_KM2)
    }
}

impl RiverThresholdPolicy {
    pub const fn legacy() -> Self {
        Self::Legacy {
            all_fraction: 0.00005,
            outlet_fraction: 0.004,
            branch_fraction: 0.0006,
        }
    }

    pub const fn catchment(minimum: f32) -> Self {
        Self::CatchmentKm2 {
            minimum,
            major_outlet_multiplier: 75.0,
            major_branch_multiplier: 12.5,
        }
    }

    /// Effective minimum catchment represented by the `all_cells` mask.
    ///
    /// `None` denotes the legacy count-equivalent policy. A catchment policy is
    /// physical only once its requested scale spans at least four global-mean
    /// cells; below that point the semantic network is resolution-limited.
    pub fn effective_all_minimum_km2(self, num_cells: usize) -> Option<f32> {
        match self {
            Self::Legacy { .. } => None,
            Self::CatchmentKm2 { minimum, .. } => {
                let mean_cell_km2 =
                    4.0 * std::f32::consts::PI * PLANET_RADIUS_KM.powi(2) / num_cells.max(1) as f32;
                Some(minimum.max(4.0 * mean_cell_km2))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct WaterBodyId {
    pub basin_id: Option<usize>,
    pub anchor_cell: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SemanticWaterKind {
    Ocean,
    Lake,
    Pond,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind", content = "target")]
pub enum WaterOutlet {
    Ocean,
    Basin(usize),
    Terminal,
}

#[derive(Clone, Debug, Serialize)]
pub struct SemanticWaterBody {
    pub id: WaterBodyId,
    pub kind: SemanticWaterKind,
    pub cells: Vec<usize>,
    pub area_km2: f32,
    pub surface_elevation_km: f32,
    pub max_depth_km: f32,
    pub outlet: WaterOutlet,
}

#[derive(Clone, Debug, Serialize)]
pub struct WaterBodySemantics {
    pub bodies: Vec<SemanticWaterBody>,
    pub cell_body: Vec<Option<usize>>,
}

impl WaterBodySemantics {
    pub fn build(tessellation: &Tessellation, hydrology: &Hydrology) -> Self {
        let n = tessellation.num_cells();
        let areas = tessellation.cell_areas();
        let mut bodies = Vec::new();
        let mut cell_body = vec![None; n];

        // Oceans are semantic connected components rather than one assumed sea.
        let mut visited = vec![false; n];
        for start in 0..n {
            if !hydrology.is_ocean(start) || visited[start] {
                continue;
            }
            let mut cells = Vec::new();
            let mut queue = VecDeque::from([start]);
            visited[start] = true;
            while let Some(cell) = queue.pop_front() {
                cells.push(cell);
                for &next in tessellation.neighbors(cell) {
                    if hydrology.is_ocean(next) && !visited[next] {
                        visited[next] = true;
                        queue.push_back(next);
                    }
                }
            }
            let anchor_cell = deepest_cell(&cells, &hydrology.elevation);
            let body_idx = bodies.len();
            for &cell in &cells {
                cell_body[cell] = Some(body_idx);
            }
            bodies.push(SemanticWaterBody {
                id: WaterBodyId {
                    basin_id: None,
                    anchor_cell,
                },
                kind: SemanticWaterKind::Ocean,
                area_km2: cells
                    .iter()
                    .map(|&cell| solid_angle_to_km2(areas[cell]))
                    .sum(),
                surface_elevation_km: 0.0,
                max_depth_km: elevation_to_km(-hydrology.elevation[anchor_cell]).max(0.0),
                outlet: WaterOutlet::Terminal,
                cells,
            });
        }

        for water_body in &hydrology.water_bodies {
            if water_body.cells.is_empty() {
                continue;
            }
            let basin = &hydrology.basins[water_body.basin_id];
            let anchor_cell = deepest_cell(&water_body.cells, &hydrology.elevation);
            let body_idx = bodies.len();
            for &cell in &water_body.cells {
                cell_body[cell] = Some(body_idx);
            }
            let outlet = if basin.is_overflowing() {
                basin
                    .overflow_target
                    .map(WaterOutlet::Basin)
                    .unwrap_or(WaterOutlet::Ocean)
            } else {
                WaterOutlet::Terminal
            };
            bodies.push(SemanticWaterBody {
                id: WaterBodyId {
                    basin_id: Some(water_body.basin_id),
                    anchor_cell,
                },
                kind: if water_body.is_lake {
                    SemanticWaterKind::Lake
                } else {
                    SemanticWaterKind::Pond
                },
                area_km2: water_body
                    .cells
                    .iter()
                    .map(|&cell| solid_angle_to_km2(areas[cell]))
                    .sum(),
                surface_elevation_km: elevation_to_km(basin.water_level),
                max_depth_km: elevation_to_km(water_body.max_depth),
                outlet,
                cells: water_body.cells.clone(),
            });
        }

        Self { bodies, cell_body }
    }
}

fn deepest_cell(cells: &[usize], elevation: &[f32]) -> usize {
    cells
        .iter()
        .copied()
        .min_by(|&a, &b| {
            elevation[a]
                .total_cmp(&elevation[b])
                .then_with(|| a.cmp(&b))
        })
        .expect("semantic water body must contain a cell")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind", content = "water_body")]
pub enum RiverMouth {
    Ocean(usize),
    Lake(usize),
    Inland,
    Confluence,
}

#[derive(Clone, Debug, Serialize)]
pub struct RiverReach {
    pub id: usize,
    /// Ordered upstream to downstream; confluence cells may be shared by reaches.
    pub cells: Vec<usize>,
    pub strahler_order: u8,
    pub is_major: bool,
    pub length_km_approx: f32,
    pub downstream: RiverMouth,
}

#[derive(Clone, Debug, Serialize)]
pub struct RiverSelection {
    pub policy: RiverThresholdPolicy,
    pub all_cells: Vec<bool>,
    pub major_cells: Vec<bool>,
    pub max_flow: f32,
    pub max_flow_count_equivalent: f32,
    pub lake_outflow_paths: Vec<(usize, Vec<usize>)>,
}

impl RiverSelection {
    /// Build only the semantic visibility/importance masks needed by rendering.
    /// This avoids allocating the full upstream graph and reach catalog.
    pub fn build(hydrology: &Hydrology, policy: RiverThresholdPolicy) -> Self {
        let n = hydrology.drainage_dir.len();
        let (all_flow, outlet_count, branch_count) = thresholds(hydrology, n, policy);
        let all_cells: Vec<bool> = (0..n)
            .map(|cell| {
                hydrology.flow_accumulation[cell] >= all_flow && !hydrology.is_submerged(cell)
            })
            .collect();
        let major_cells: Vec<bool> = hydrology
            .compute_major_river_cells(outlet_count, branch_count)
            .into_iter()
            .zip(&all_cells)
            .map(|(major, &all)| major && all)
            .collect();
        let max_flow = hydrology
            .flow_accumulation
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let max_flow_count_equivalent = (0..n)
            .map(|cell| hydrology.flow_count_equiv(cell))
            .fold(0.0f32, f32::max);
        Self {
            policy,
            all_cells,
            major_cells,
            max_flow,
            max_flow_count_equivalent,
            lake_outflow_paths: hydrology.lake_outflow_paths(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RiverNetwork {
    pub policy: RiverThresholdPolicy,
    pub all_cells: Vec<bool>,
    pub major_cells: Vec<bool>,
    pub upstream: Vec<Vec<usize>>,
    pub strahler_order: Vec<u8>,
    pub reaches: Vec<RiverReach>,
    pub mouths: Vec<usize>,
    pub max_flow: f32,
    pub max_flow_count_equivalent: f32,
    pub lake_outflow_paths: Vec<(usize, Vec<usize>)>,
}

impl RiverNetwork {
    pub fn build(
        tessellation: &Tessellation,
        hydrology: &Hydrology,
        water: &WaterBodySemantics,
        policy: RiverThresholdPolicy,
    ) -> Self {
        let n = tessellation.num_cells();
        let selection = RiverSelection::build(hydrology, policy);
        let all_cells = selection.all_cells;
        let major_cells = selection.major_cells;

        let mut upstream = vec![Vec::new(); n];
        for cell in 0..n {
            if !all_cells[cell] {
                continue;
            }
            if let Some(downstream) = hydrology.downstream(cell) {
                if all_cells[downstream] {
                    upstream[downstream].push(cell);
                }
            }
        }
        for cells in &mut upstream {
            cells.sort_unstable();
        }

        let mut channel_cells: Vec<usize> = (0..n).filter(|&cell| all_cells[cell]).collect();
        channel_cells.sort_by(|&a, &b| {
            hydrology.flow_accumulation[a]
                .total_cmp(&hydrology.flow_accumulation[b])
                .then_with(|| a.cmp(&b))
        });
        let mut strahler_order = vec![0u8; n];
        for &cell in &channel_cells {
            let (mut best, mut ties) = (0u8, 0usize);
            for &up in &upstream[cell] {
                let order = strahler_order[up];
                if order > best {
                    best = order;
                    ties = 1;
                } else if order == best {
                    ties += 1;
                }
            }
            strahler_order[cell] = if best == 0 {
                1
            } else if ties >= 2 {
                best + 1
            } else {
                best
            };
        }

        let mouths: Vec<usize> = channel_cells
            .iter()
            .copied()
            .filter(|&cell| match hydrology.downstream(cell) {
                Some(next) => !all_cells[next] && hydrology.is_submerged(next),
                None => true,
            })
            .collect();
        let reaches = build_reaches(
            tessellation,
            hydrology,
            water,
            &all_cells,
            &major_cells,
            &upstream,
            &strahler_order,
        );

        Self {
            policy,
            all_cells,
            major_cells,
            upstream,
            strahler_order,
            reaches,
            mouths,
            max_flow: selection.max_flow,
            max_flow_count_equivalent: selection.max_flow_count_equivalent,
            lake_outflow_paths: selection.lake_outflow_paths,
        }
    }
}

fn thresholds(
    hydrology: &Hydrology,
    num_cells: usize,
    policy: RiverThresholdPolicy,
) -> (f32, f32, f32) {
    match policy {
        RiverThresholdPolicy::Legacy {
            all_fraction,
            outlet_fraction,
            branch_fraction,
        } => {
            let count = num_cells.max(1) as f32;
            (
                (count * all_fraction).max(1.0) * hydrology.mean_cell_discharge,
                (count * outlet_fraction).max(1.0),
                (count * branch_fraction).max(1.0),
            )
        }
        RiverThresholdPolicy::CatchmentKm2 {
            minimum,
            major_outlet_multiplier,
            major_branch_multiplier,
        } => {
            let all_minimum = policy
                .effective_all_minimum_km2(num_cells)
                .expect("catchment policy has an effective physical threshold");
            let per_count = hydrology.mean_cell_discharge.max(1e-12);
            (
                Hydrology::flow_for_catchment_km2(all_minimum),
                Hydrology::flow_for_catchment_km2(major_outlet_multiplier * minimum) / per_count,
                Hydrology::flow_for_catchment_km2(major_branch_multiplier * minimum) / per_count,
            )
        }
    }
}

#[cfg(test)]
mod policy_tests {
    use super::RiverThresholdPolicy;

    #[test]
    fn catchment_policy_discloses_resolution_floor() {
        let policy = RiverThresholdPolicy::catchment(2_000.0);
        let at_one_million = policy.effective_all_minimum_km2(1_000_000).unwrap();
        let at_one_hundred_million = policy.effective_all_minimum_km2(100_000_000).unwrap();

        assert!(at_one_million > 2_000.0);
        assert!((at_one_million - 2_040.0).abs() < 5.0);
        assert_eq!(at_one_hundred_million, 2_000.0);
        assert_eq!(
            RiverThresholdPolicy::legacy().effective_all_minimum_km2(1_000_000),
            None
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn build_reaches(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    water: &WaterBodySemantics,
    all_cells: &[bool],
    major_cells: &[bool],
    upstream: &[Vec<usize>],
    strahler_order: &[u8],
) -> Vec<RiverReach> {
    let mut starts: Vec<usize> = (0..all_cells.len())
        .filter(|&cell| all_cells[cell] && upstream[cell].len() != 1)
        .collect();
    starts.sort_unstable();
    let mut reaches = Vec::new();
    for start in starts {
        let mut cells = vec![start];
        let mut current = start;
        let downstream = loop {
            let Some(next) = hydrology.downstream(current) else {
                break RiverMouth::Inland;
            };
            if !all_cells[next] {
                if hydrology.is_ocean(next) {
                    break RiverMouth::Ocean(
                        water.cell_body[next].expect("ocean cell has semantic water body"),
                    );
                }
                if hydrology.is_lake_water(next) {
                    break RiverMouth::Lake(
                        water.cell_body[next].expect("lake cell has semantic water body"),
                    );
                }
                break RiverMouth::Inland;
            }
            cells.push(next);
            current = next;
            if upstream[current].len() != 1 {
                break RiverMouth::Confluence;
            }
        };
        let length_km_approx = cells
            .windows(2)
            .map(|pair| {
                (tessellation.cell_center(pair[0]) - tessellation.cell_center(pair[1])).length()
                    * PLANET_RADIUS_KM
            })
            .sum();
        reaches.push(RiverReach {
            id: reaches.len(),
            strahler_order: strahler_order[start],
            is_major: cells.iter().any(|&cell| major_cells[cell]),
            length_km_approx,
            downstream,
            cells,
        });
    }
    reaches
}
