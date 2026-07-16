//! Exact categorical shoreline geometry derived from retained hydrology.
//!
//! These loops are boundaries of ocean/lake Voronoi-cell masks. They are not a
//! zero-elevation contour and are deliberately not simplified or styled for a
//! map. Vertex IDs reference the source tessellation, keeping the semantic
//! object compact and independent of camera and relief exaggeration.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::Serialize;

use super::{
    Hydrology, SemanticWaterKind, Tessellation, WaterBodyId, WaterBodySemantics, PLANET_RADIUS_KM,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DirectedShorelineEdge {
    /// Directed so the owning water body is on the left.
    pub from_vertex: u32,
    pub to_vertex: u32,
    pub water_cell: usize,
    pub land_cell: usize,
    pub landmass_anchor_cell: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ShorelineLoop {
    pub water_body_id: WaterBodyId,
    pub water_kind: SemanticWaterKind,
    /// Stable only within this retained tessellation.
    pub anchor_edge: [u32; 2],
    pub edges: Vec<DirectedShorelineEdge>,
    pub length_km: f32,
    pub adjacent_landmass_anchor_cells: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ShorelineTopologyIssue {
    pub water_body_id: WaterBodyId,
    pub vertex_index: u32,
    pub incoming_edge_count: usize,
    pub outgoing_edge_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct UnresolvedShorelineEdge {
    pub water_body_id: WaterBodyId,
    pub water_kind: SemanticWaterKind,
    pub edge: DirectedShorelineEdge,
}

/// On-demand raw geometry; intentionally excluded from the compact dossier.
#[derive(Clone, Debug, Serialize)]
pub struct ShorelineGeometry {
    pub landmass_anchor_by_cell: Vec<Option<usize>>,
    pub loops: Vec<ShorelineLoop>,
    pub unresolved_edges: Vec<UnresolvedShorelineEdge>,
    pub issues: Vec<ShorelineTopologyIssue>,
}

#[derive(Clone, Copy)]
struct OwnedEdge {
    body_id: WaterBodyId,
    water_kind: SemanticWaterKind,
    edge: DirectedShorelineEdge,
}

impl ShorelineGeometry {
    pub fn build(
        tessellation: &Tessellation,
        hydrology: &Hydrology,
        water: &WaterBodySemantics,
    ) -> Result<Self, String> {
        let n = tessellation.num_cells();
        if hydrology.elevation.len() != n || water.cell_body.len() != n {
            return Err("shoreline inputs do not match tessellation cell count".to_string());
        }

        let landmass_anchor_by_cell = landmass_anchors(tessellation, hydrology);
        let mut owned_edges = Vec::new();
        for (land_cell, &landmass_anchor) in landmass_anchor_by_cell.iter().enumerate() {
            let Some(landmass_anchor_cell) = landmass_anchor else {
                continue;
            };
            for &water_cell in tessellation.neighbors(land_cell) {
                if !hydrology.is_submerged(water_cell) {
                    continue;
                }
                let body_index = water.cell_body[water_cell].ok_or_else(|| {
                    format!("submerged shoreline cell {water_cell} has no semantic owner")
                })?;
                let body = water.bodies.get(body_index).ok_or_else(|| {
                    format!("shoreline cell {water_cell} has invalid semantic owner {body_index}")
                })?;
                if body.kind == SemanticWaterKind::Pond {
                    return Err(format!(
                        "submerged shoreline cell {water_cell} is owned by a semantic pond"
                    ));
                }
                let [land_from, land_to] = tessellation
                    .shared_edge_vertices(land_cell, water_cell)
                    .ok_or_else(|| {
                        format!("adjacent cells {land_cell} and {water_cell} lack a shared edge")
                    })?;
                // Reverse the land cell's CCW half-edge: the water body is then left.
                owned_edges.push(OwnedEdge {
                    body_id: body.id,
                    water_kind: body.kind,
                    edge: DirectedShorelineEdge {
                        from_vertex: land_to,
                        to_vertex: land_from,
                        water_cell,
                        land_cell,
                        landmass_anchor_cell,
                    },
                });
            }
        }
        owned_edges.sort_by_key(edge_key);

        let mut grouped: BTreeMap<(Option<usize>, usize), Vec<OwnedEdge>> = BTreeMap::new();
        for edge in owned_edges {
            grouped
                .entry((edge.body_id.basin_id, edge.body_id.anchor_cell))
                .or_default()
                .push(edge);
        }

        let mut loops = Vec::new();
        let mut unresolved_edges = Vec::new();
        let mut issues = Vec::new();
        for edges in grouped.values() {
            trace_body_loops(
                tessellation,
                edges,
                &mut loops,
                &mut unresolved_edges,
                &mut issues,
            );
        }
        loops.sort_by_key(|shore| {
            (
                shore.water_body_id.basin_id,
                shore.water_body_id.anchor_cell,
                shore.anchor_edge,
            )
        });
        unresolved_edges.sort_by_key(|edge| {
            (
                edge.water_body_id.basin_id,
                edge.water_body_id.anchor_cell,
                edge.edge.from_vertex,
                edge.edge.to_vertex,
                edge.edge.land_cell,
            )
        });
        issues.sort_by_key(|issue| {
            (
                issue.water_body_id.basin_id,
                issue.water_body_id.anchor_cell,
                issue.vertex_index,
            )
        });

        Ok(Self {
            landmass_anchor_by_cell,
            loops,
            unresolved_edges,
            issues,
        })
    }
}

fn edge_key(edge: &OwnedEdge) -> (Option<usize>, usize, u32, u32, usize, usize) {
    (
        edge.body_id.basin_id,
        edge.body_id.anchor_cell,
        edge.edge.from_vertex,
        edge.edge.to_vertex,
        edge.edge.land_cell,
        edge.edge.water_cell,
    )
}

fn landmass_anchors(tessellation: &Tessellation, hydrology: &Hydrology) -> Vec<Option<usize>> {
    let n = tessellation.num_cells();
    let mut owners = vec![None; n];
    for start in 0..n {
        if hydrology.is_submerged(start) || owners[start].is_some() {
            continue;
        }
        let mut members = Vec::new();
        let mut queue = VecDeque::from([start]);
        owners[start] = Some(start);
        while let Some(cell) = queue.pop_front() {
            members.push(cell);
            for &next in tessellation.neighbors(cell) {
                if !hydrology.is_submerged(next) && owners[next].is_none() {
                    owners[next] = Some(start);
                    queue.push_back(next);
                }
            }
        }
        let anchor = *members.iter().min().expect("land component contains start");
        for cell in members {
            owners[cell] = Some(anchor);
        }
    }
    owners
}

fn trace_body_loops(
    tessellation: &Tessellation,
    edges: &[OwnedEdge],
    loops: &mut Vec<ShorelineLoop>,
    unresolved: &mut Vec<UnresolvedShorelineEdge>,
    issues: &mut Vec<ShorelineTopologyIssue>,
) {
    let body_id = edges[0].body_id;
    let water_kind = edges[0].water_kind;
    let mut incoming: BTreeMap<u32, usize> = BTreeMap::new();
    let mut outgoing: BTreeMap<u32, usize> = BTreeMap::new();
    for owned in edges {
        *outgoing.entry(owned.edge.from_vertex).or_default() += 1;
        *incoming.entry(owned.edge.to_vertex).or_default() += 1;
    }
    let vertices: BTreeSet<u32> = incoming.keys().chain(outgoing.keys()).copied().collect();
    let mut bad_vertices = BTreeSet::new();
    for vertex in vertices {
        let inc = incoming.get(&vertex).copied().unwrap_or(0);
        let out = outgoing.get(&vertex).copied().unwrap_or(0);
        if inc != 1 || out != 1 {
            bad_vertices.insert(vertex);
            issues.push(ShorelineTopologyIssue {
                water_body_id: body_id,
                vertex_index: vertex,
                incoming_edge_count: inc,
                outgoing_edge_count: out,
            });
        }
    }

    let mut by_start = BTreeMap::new();
    let mut remaining = BTreeSet::new();
    for (index, owned) in edges.iter().enumerate() {
        if bad_vertices.contains(&owned.edge.from_vertex)
            || bad_vertices.contains(&owned.edge.to_vertex)
            || by_start.insert(owned.edge.from_vertex, index).is_some()
        {
            unresolved.push(UnresolvedShorelineEdge {
                water_body_id: body_id,
                water_kind,
                edge: owned.edge,
            });
        } else {
            remaining.insert(index);
        }
    }

    while let Some(&start_index) = remaining.iter().next() {
        let start_vertex = edges[start_index].edge.from_vertex;
        let mut current = start_index;
        let mut cycle = Vec::new();
        let mut closed = false;
        loop {
            if !remaining.remove(&current) {
                break;
            }
            let edge = edges[current].edge;
            cycle.push(edge);
            if edge.to_vertex == start_vertex {
                closed = true;
                break;
            }
            let Some(&next) = by_start.get(&edge.to_vertex) else {
                break;
            };
            current = next;
        }

        if closed && cycle.len() >= 3 {
            let mut anchors: Vec<usize> =
                cycle.iter().map(|edge| edge.landmass_anchor_cell).collect();
            anchors.sort_unstable();
            anchors.dedup();
            let length_km = cycle
                .iter()
                .map(|edge| {
                    let from = tessellation.voronoi.vertices[edge.from_vertex as usize];
                    let to = tessellation.voronoi.vertices[edge.to_vertex as usize];
                    (from - to).length() * PLANET_RADIUS_KM
                })
                .sum();
            loops.push(ShorelineLoop {
                water_body_id: body_id,
                water_kind,
                anchor_edge: [cycle[0].from_vertex, cycle[0].to_vertex],
                edges: cycle,
                length_km,
                adjacent_landmass_anchor_cells: anchors,
            });
        } else {
            unresolved.extend(cycle.into_iter().map(|edge| UnresolvedShorelineEdge {
                water_body_id: body_id,
                water_kind,
                edge,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::world::{Elevation, NoiseLayerData};

    #[test]
    fn generated_hemisphere_has_closed_exact_shoreline() {
        let mut rng = ChaCha8Rng::seed_from_u64(73);
        let tessellation = Tessellation::generate(400, 0, &mut rng);
        let n = tessellation.num_cells();
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
        let geometry = ShorelineGeometry::build(&tessellation, &hydrology, &water).unwrap();

        assert!(!geometry.loops.is_empty());
        assert!(
            geometry.unresolved_edges.is_empty(),
            "{:?}",
            geometry.issues
        );
        assert!(geometry.issues.is_empty());
        for shoreline in &geometry.loops {
            assert!(shoreline.edges.len() >= 3);
            for pair in shoreline.edges.windows(2) {
                assert_eq!(pair[0].to_vertex, pair[1].from_vertex);
            }
            assert_eq!(
                shoreline.edges.last().unwrap().to_vertex,
                shoreline.edges[0].from_vertex
            );
            assert!(shoreline.edges.iter().all(|edge| {
                hydrology.is_submerged(edge.water_cell) && !hydrology.is_submerged(edge.land_cell)
            }));
        }

        let interface_edges = (0..n)
            .filter(|&cell| !hydrology.is_submerged(cell))
            .map(|cell| {
                tessellation
                    .neighbors(cell)
                    .iter()
                    .filter(|&&next| hydrology.is_submerged(next))
                    .count()
            })
            .sum::<usize>();
        assert_eq!(
            geometry
                .loops
                .iter()
                .map(|shoreline| shoreline.edges.len())
                .sum::<usize>(),
            interface_edges
        );
    }
}
