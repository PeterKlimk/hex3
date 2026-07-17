//! Bounded terrestrial route evidence for Consequential Geography V0.
//!
//! Routes are an on-demand derivative over one aggregate-site selection. They
//! use the authoritative Stage-4 land graph and physical terrain, form a sparse
//! candidate graph without crossing water, and retain all bounded candidate
//! paths needed for counterfactual comparison. This is not a general routing
//! service and does not imply roads, travel time, population, or maritime links.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};

use serde::Serialize;

use super::{
    elevation_to_km, AggregateSiteSelection, Hydrology, Tessellation, TraversalConfig,
    PLANET_RADIUS_KM,
};

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RouteNetworkConfig {
    /// Per-site physical nearest neighbors added beside the physical spanning
    /// tree for each occupied landmass.
    pub nearest_neighbors_per_site: usize,
    /// Hard cap before any terrain searches run.
    pub maximum_candidate_pair_count: usize,
    /// Hard cap on cells settled across all multi-target searches.
    pub maximum_total_search_cell_visits: usize,
    /// Global cap on links added beyond the generalized-cost spanning forest.
    pub maximum_extra_links: usize,
    /// Add an available link only when existing-network cost divided by direct
    /// route cost reaches this ratio.
    pub minimum_extra_link_detour_ratio: f32,
}

impl RouteNetworkConfig {
    pub fn validate(self) -> Result<Self, &'static str> {
        if !(1..=6).contains(&self.nearest_neighbors_per_site) {
            return Err("route nearest-neighbor count must be within 1..=6");
        }
        if self.maximum_candidate_pair_count == 0 || self.maximum_candidate_pair_count > 128 {
            return Err("route candidate-pair cap must be within 1..=128");
        }
        if self.maximum_total_search_cell_visits == 0
            || self.maximum_total_search_cell_visits > 20_000_000
        {
            return Err("route settled-cell budget must be within 1..=20,000,000");
        }
        if self.maximum_extra_links > 8 {
            return Err("route extra-link cap must be at most 8");
        }
        if !self.minimum_extra_link_detour_ratio.is_finite()
            || self.minimum_extra_link_detour_ratio < 1.0
        {
            return Err("route detour ratio must be finite and at least 1");
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouteSelectionRole {
    Backbone,
    DetourReduction,
}

#[derive(Clone, Debug, Serialize)]
pub struct AggregateRouteCandidate {
    pub id: usize,
    pub from_site_id: usize,
    pub to_site_id: usize,
    pub landmass_anchor_cell: usize,
    pub ordered_cells: Vec<usize>,
    pub physical_length_km: f32,
    pub symmetric_generalized_cost_km: f32,
    pub forward_generalized_cost_km: f32,
    pub reverse_generalized_cost_km: f32,
    pub ascent_km_from_from_site: f32,
    pub descent_km_from_from_site: f32,
    pub maximum_absolute_grade: f32,
    pub drainage_repaired_cell_count: usize,
    pub drainage_repaired_edge_count: usize,
    /// Full physical length of every path edge for which either endpoint was
    /// lowered by drainage integration; this is support attribution, not a
    /// geometric overlap length within an edge.
    pub drainage_repaired_length_km: f32,
    pub drainage_repaired_length_fraction: f32,
    pub selection_role: Option<RouteSelectionRole>,
    pub network_detour_before_generalized_km: Option<f32>,
    pub detour_ratio_before: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TerrestrialRouteComponent {
    pub landmass_anchor_cell: usize,
    pub site_ids: Vec<usize>,
    pub candidate_route_ids: Vec<usize>,
    pub selected_route_ids: Vec<usize>,
    pub backbone_route_count: usize,
    pub extra_route_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct AggregateRouteNetwork {
    pub config: RouteNetworkConfig,
    pub traversal: TraversalConfig,
    pub site_anchor_cells: Vec<usize>,
    pub occupied_landmass_count: usize,
    pub routable_landmass_count: usize,
    pub isolated_site_ids: Vec<usize>,
    pub candidate_pair_count: usize,
    pub route_search_count: usize,
    pub search_settled_cell_count: usize,
    pub backbone_route_count: usize,
    pub extra_route_count: usize,
    pub selected_route_ids: Vec<usize>,
    pub components: Vec<TerrestrialRouteComponent>,
    /// Every bounded candidate path is retained so fixed-endpoint physical and
    /// zero-grade counterfactuals can be compared without rerouting all pairs.
    pub candidate_routes: Vec<AggregateRouteCandidate>,
    pub exact_tie_policy: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum RouteLowerCorridorOmissionReason {
    NoTerrainBurden,
    ExactPath,
    NonMonotonicCommonCellOrder,
    NoSimpleDivergence,
    NoDistanceTerrainTradeoff,
    NoGeneralizedCostAdvantage,
    NoLowerCrossing,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouteLowerCorridorOmission {
    pub candidate_route_id: usize,
    pub endpoint_site_ids: [usize; 2],
    pub physical_selection_role: Option<RouteSelectionRole>,
    pub distance_null_selection_role: Option<RouteSelectionRole>,
    pub reason: RouteLowerCorridorOmissionReason,
    pub divergent_segment_count: usize,
    pub distance_terrain_tradeoff_count: usize,
    pub generalized_cost_advantage_count: usize,
    pub lower_crossing_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouteSegmentTerrainEvidence {
    pub edge_count: usize,
    pub physical_length_km: f32,
    /// Rescored under the physical network's traversal configuration in both
    /// branches, including for the distance-null path.
    pub symmetric_generalized_cost_km: f32,
    pub ascent_km: f32,
    pub descent_km: f32,
    pub maximum_elevation_km: f32,
    pub maximum_elevation_cell: usize,
    pub maximum_absolute_grade: f32,
    pub drainage_repaired_cell_count: usize,
    pub drainage_repaired_edge_count: usize,
    /// Full length of edges supported by at least one repaired endpoint, not
    /// sub-edge geometric overlap with a carved feature.
    pub drainage_repaired_length_km: f32,
    pub drainage_repaired_length_fraction: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouteLowerCorridorEvidence {
    pub candidate_route_id: usize,
    pub endpoint_site_ids: [usize; 2],
    pub physical_selection_role: Option<RouteSelectionRole>,
    pub distance_null_selection_role: Option<RouteSelectionRole>,
    pub divergence_endpoint_cells: [usize; 2],
    /// Inclusive indices into the corresponding candidate's `ordered_cells`.
    pub physical_segment_cell_indices: [usize; 2],
    pub distance_null_segment_cell_indices: [usize; 2],
    pub physical_segment: RouteSegmentTerrainEvidence,
    pub distance_null_segment: RouteSegmentTerrainEvidence,
    pub physical_length_premium_km: f32,
    pub generalized_cost_saved_km: f32,
    pub generalized_cost_saved_fraction: f32,
    pub ascent_saved_km: f32,
    pub maximum_elevation_saved_km: f32,
    pub divergence_endpoint_distance_km: f32,
    /// Symmetric Hausdorff-like separation sampled only at retained path cell
    /// centers; it is not continuous curve-to-curve or corridor width.
    pub maximum_symmetric_cell_center_separation_km: f32,
    /// Cells `[physical, distance-null]` attaining the sampled separation,
    /// with deterministic cell-ID tie-breaking.
    pub maximum_symmetric_cell_center_separation_cells: [usize; 2],
    pub whole_route_divergence: bool,
    pub interpretation: &'static str,
    pub limitation: &'static str,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case")]
pub enum RouteLowerCorridorAssessment {
    Explained {
        evidence: Box<RouteLowerCorridorEvidence>,
    },
    Omitted {
        omission: RouteLowerCorridorOmission,
    },
}

#[derive(Clone, Copy, Debug)]
struct SiteAnchor {
    id: usize,
    cell: usize,
}

#[derive(Clone, Copy, Debug)]
struct EdgeMetrics {
    distance_km: f32,
    forward_cost_km: f32,
    reverse_cost_km: f32,
    symmetric_cost_km: f32,
    ascent_km: f32,
    descent_km: f32,
    absolute_grade: f32,
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

/// Build one bounded terrestrial route network over an existing site selection.
///
/// `traversal` is explicit rather than inherited from `selection`: ordinary
/// callers should pass the selection traversal, while fixed-site diagnostic
/// counterfactuals may deliberately vary it and must disclose that mismatch.
pub fn build_aggregate_route_network(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    selection: &AggregateSiteSelection,
    traversal: TraversalConfig,
    config: RouteNetworkConfig,
) -> Result<AggregateRouteNetwork, &'static str> {
    let config = config.validate()?;
    let n = tessellation.num_cells();
    if hydrology.elevation.len() != n
        || hydrology.is_ocean.len() != n
        || hydrology.basin_id.len() != n
        || hydrology.cell_water_body.len() != n
    {
        return Err("route-network hydrology must match tessellation");
    }
    let mut sites = Vec::with_capacity(selection.sites.len());
    let mut anchors = BTreeSet::new();
    for (index, site) in selection.sites.iter().enumerate() {
        if site.id != index || site.anchor_cell >= n || hydrology.is_submerged(site.anchor_cell) {
            return Err("route sites must have contiguous IDs and in-range land anchors");
        }
        if !anchors.insert(site.anchor_cell) {
            return Err("route site anchors must be unique");
        }
        sites.push(SiteAnchor {
            id: site.id,
            cell: site.anchor_cell,
        });
    }
    let submerged: Vec<bool> = (0..n).map(|cell| hydrology.is_submerged(cell)).collect();
    let repaired: Vec<bool> = (0..n)
        .map(|cell| hydrology.was_lowered_by_integration(cell))
        .collect();
    build_route_network_from_fields(
        tessellation,
        &hydrology.elevation,
        &submerged,
        &repaired,
        &sites,
        traversal,
        config,
    )
}

/// Assess whether one physical least-cost candidate uses a demonstrably lower
/// terrain corridor than its matched physical-distance-null candidate.
///
/// This relationship is deliberately narrower than a geomorphic gap or pass:
/// it compares two bounded paths through the same effective Stage-4 terrain
/// but has no independent ridge, barrier, or chokepoint topology.
pub fn assess_route_lower_corridor(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    physical: &AggregateRouteNetwork,
    distance_null: &AggregateRouteNetwork,
    candidate_route_id: usize,
) -> Result<RouteLowerCorridorAssessment, &'static str> {
    let n = tessellation.num_cells();
    if hydrology.elevation.len() != n
        || hydrology.is_ocean.len() != n
        || hydrology.basin_id.len() != n
        || hydrology.cell_water_body.len() != n
    {
        return Err("route-corridor hydrology must match tessellation");
    }
    let is_submerged = |cell| hydrology.is_submerged(cell);
    let is_repaired = |cell| hydrology.was_lowered_by_integration(cell);
    assess_route_lower_corridor_from_fields(
        tessellation,
        &hydrology.elevation,
        &is_submerged,
        &is_repaired,
        physical,
        distance_null,
        candidate_route_id,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_route_network_from_fields(
    tessellation: &Tessellation,
    elevation: &[f32],
    submerged: &[bool],
    repaired: &[bool],
    sites: &[SiteAnchor],
    traversal: TraversalConfig,
    config: RouteNetworkConfig,
) -> Result<AggregateRouteNetwork, &'static str> {
    let n = tessellation.num_cells();
    if elevation.len() != n || submerged.len() != n || repaired.len() != n {
        return Err("route fields must match tessellation");
    }
    if elevation.iter().any(|value| !value.is_finite()) {
        return Err("route terrain must be finite");
    }
    for (index, site) in sites.iter().enumerate() {
        if site.id != index || site.cell >= n || submerged[site.cell] {
            return Err("route field sites must have contiguous IDs and land anchors");
        }
    }
    let owners = landmass_anchors(tessellation, submerged);
    let mut occupied: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for site in sites {
        let anchor = owners[site.cell].ok_or("route site has no terrestrial component")?;
        occupied.entry(anchor).or_default().push(site.id);
    }

    let mut isolated_site_ids = Vec::new();
    let mut candidate_pairs = BTreeSet::new();
    for site_ids in occupied.values() {
        if site_ids.len() == 1 {
            isolated_site_ids.push(site_ids[0]);
            continue;
        }
        add_physical_mst_pairs(tessellation, sites, site_ids, &mut candidate_pairs);
        add_nearest_neighbor_pairs(
            tessellation,
            sites,
            site_ids,
            config.nearest_neighbors_per_site,
            &mut candidate_pairs,
        );
    }
    if candidate_pairs.len() > config.maximum_candidate_pair_count {
        return Err("route candidate graph exceeds configured pair cap");
    }
    let pairs: Vec<(usize, usize)> = candidate_pairs.into_iter().collect();
    let (paths, route_search_count, search_settled_cell_count) = route_candidate_paths(
        tessellation,
        elevation,
        submerged,
        sites,
        &pairs,
        traversal,
        config.maximum_total_search_cell_visits,
    )?;
    let mut candidate_routes = Vec::with_capacity(pairs.len());
    for (id, ((from_site_id, to_site_id), path)) in pairs.iter().copied().zip(paths).enumerate() {
        let component =
            owners[sites[from_site_id].cell].ok_or("routed site lost terrestrial ownership")?;
        if owners[sites[to_site_id].cell] != Some(component) {
            return Err("route candidate crosses terrestrial components");
        }
        candidate_routes.push(route_metrics(
            id,
            from_site_id,
            to_site_id,
            component,
            path,
            tessellation,
            elevation,
            repaired,
            traversal,
        )?);
    }

    let mut selected_route_ids = Vec::new();
    for site_ids in occupied.values().filter(|ids| ids.len() > 1) {
        let mut routes: Vec<usize> = candidate_routes
            .iter()
            .filter(|route| site_ids.binary_search(&route.from_site_id).is_ok())
            .map(|route| route.id)
            .collect();
        routes.sort_unstable_by(|&a, &b| {
            candidate_routes[a]
                .symmetric_generalized_cost_km
                .total_cmp(&candidate_routes[b].symmetric_generalized_cost_km)
                .then_with(|| {
                    candidate_routes[a]
                        .from_site_id
                        .cmp(&candidate_routes[b].from_site_id)
                })
                .then_with(|| {
                    candidate_routes[a]
                        .to_site_id
                        .cmp(&candidate_routes[b].to_site_id)
                })
        });
        let mut dsu = DisjointSet::new(sites.len());
        let mut count = 0usize;
        for route_id in routes {
            let route = &candidate_routes[route_id];
            if dsu.union(route.from_site_id, route.to_site_id) {
                candidate_routes[route_id].selection_role = Some(RouteSelectionRole::Backbone);
                selected_route_ids.push(route_id);
                count += 1;
                if count + 1 == site_ids.len() {
                    break;
                }
            }
        }
        if count + 1 != site_ids.len() {
            return Err("route candidate graph cannot connect an occupied landmass");
        }
    }
    let backbone_route_count = selected_route_ids.len();

    for _ in 0..config.maximum_extra_links {
        let mut best: Option<(usize, f32, f32, f32)> = None;
        for route in &candidate_routes {
            if route.selection_role.is_some() {
                continue;
            }
            let Some(detour) = selected_network_distance(
                sites.len(),
                &candidate_routes,
                &selected_route_ids,
                route.from_site_id,
                route.to_site_id,
            ) else {
                continue;
            };
            let ratio = detour / route.symmetric_generalized_cost_km;
            if ratio < config.minimum_extra_link_detour_ratio {
                continue;
            }
            let saved = detour - route.symmetric_generalized_cost_km;
            let replace = best.is_none_or(|(best_id, best_saved, _, _)| {
                saved > best_saved
                    || (saved.to_bits() == best_saved.to_bits()
                        && (route.from_site_id, route.to_site_id)
                            < (
                                candidate_routes[best_id].from_site_id,
                                candidate_routes[best_id].to_site_id,
                            ))
            });
            if replace {
                best = Some((route.id, saved, detour, ratio));
            }
        }
        let Some((route_id, _, detour, ratio)) = best else {
            break;
        };
        candidate_routes[route_id].selection_role = Some(RouteSelectionRole::DetourReduction);
        candidate_routes[route_id].network_detour_before_generalized_km = Some(detour);
        candidate_routes[route_id].detour_ratio_before = Some(ratio);
        selected_route_ids.push(route_id);
    }
    let extra_route_count = selected_route_ids.len() - backbone_route_count;

    let mut components = Vec::with_capacity(occupied.len());
    for (&landmass_anchor_cell, site_ids) in &occupied {
        let candidate_route_ids: Vec<usize> = candidate_routes
            .iter()
            .filter(|route| route.landmass_anchor_cell == landmass_anchor_cell)
            .map(|route| route.id)
            .collect();
        let component_selected: Vec<usize> = selected_route_ids
            .iter()
            .copied()
            .filter(|&id| candidate_routes[id].landmass_anchor_cell == landmass_anchor_cell)
            .collect();
        let component_backbone = component_selected
            .iter()
            .filter(|&&id| {
                candidate_routes[id].selection_role == Some(RouteSelectionRole::Backbone)
            })
            .count();
        let component_extra = component_selected.len() - component_backbone;
        components.push(TerrestrialRouteComponent {
            landmass_anchor_cell,
            site_ids: site_ids.clone(),
            candidate_route_ids,
            selected_route_ids: component_selected,
            backbone_route_count: component_backbone,
            extra_route_count: component_extra,
        });
    }
    validate_selected_components(sites.len(), &components, &candidate_routes)?;

    Ok(AggregateRouteNetwork {
        config,
        traversal,
        site_anchor_cells: sites.iter().map(|site| site.cell).collect(),
        occupied_landmass_count: occupied.len(),
        routable_landmass_count: occupied.values().filter(|ids| ids.len() > 1).count(),
        isolated_site_ids,
        candidate_pair_count: candidate_routes.len(),
        route_search_count,
        search_settled_cell_count,
        backbone_route_count,
        extra_route_count,
        selected_route_ids,
        components,
        candidate_routes,
        exact_tie_policy: "candidate physical-distance ties, generalized-cost forest ties, and extra-link savings ties break by canonical site-ID endpoint pairs; Dijkstra queue ties break by lower cell ID while strict relaxation retains the first equal-cost predecessor, so path ties remain adjacency/cell-representation dependent",
    })
}

fn landmass_anchors(tessellation: &Tessellation, submerged: &[bool]) -> Vec<Option<usize>> {
    let n = tessellation.num_cells();
    let mut owners = vec![None; n];
    for start in 0..n {
        if submerged[start] || owners[start].is_some() {
            continue;
        }
        owners[start] = Some(start);
        let mut queue = VecDeque::from([start]);
        while let Some(cell) = queue.pop_front() {
            for &neighbor in tessellation.neighbors(cell) {
                if !submerged[neighbor] && owners[neighbor].is_none() {
                    owners[neighbor] = Some(start);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    owners
}

fn add_physical_mst_pairs(
    tessellation: &Tessellation,
    sites: &[SiteAnchor],
    site_ids: &[usize],
    pairs: &mut BTreeSet<(usize, usize)>,
) {
    let mut edges = Vec::new();
    for (offset, &a) in site_ids.iter().enumerate() {
        for &b in site_ids.iter().skip(offset + 1) {
            edges.push((
                physical_distance_km(tessellation, sites[a].cell, sites[b].cell),
                a,
                b,
            ));
        }
    }
    edges.sort_unstable_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut dsu = DisjointSet::new(sites.len());
    let mut accepted = 0;
    for (_, a, b) in edges {
        if dsu.union(a, b) {
            pairs.insert((a.min(b), a.max(b)));
            accepted += 1;
            if accepted + 1 == site_ids.len() {
                break;
            }
        }
    }
}

fn add_nearest_neighbor_pairs(
    tessellation: &Tessellation,
    sites: &[SiteAnchor],
    site_ids: &[usize],
    count: usize,
    pairs: &mut BTreeSet<(usize, usize)>,
) {
    for &site in site_ids {
        let mut neighbors: Vec<(f32, usize)> = site_ids
            .iter()
            .copied()
            .filter(|&other| other != site)
            .map(|other| {
                (
                    physical_distance_km(tessellation, sites[site].cell, sites[other].cell),
                    other,
                )
            })
            .collect();
        neighbors.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (_, other) in neighbors.into_iter().take(count) {
            pairs.insert((site.min(other), site.max(other)));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn route_candidate_paths(
    tessellation: &Tessellation,
    elevation: &[f32],
    submerged: &[bool],
    sites: &[SiteAnchor],
    pairs: &[(usize, usize)],
    traversal: TraversalConfig,
    maximum_visits: usize,
) -> Result<(Vec<Vec<usize>>, usize, usize), &'static str> {
    let n = tessellation.num_cells();
    let mut by_source: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (pair_index, &(from, to)) in pairs.iter().enumerate() {
        by_source.entry(from).or_default().push((to, pair_index));
    }
    let mut paths: Vec<Option<Vec<usize>>> = vec![None; pairs.len()];
    let mut distance = vec![f32::INFINITY; n];
    let mut predecessor = vec![usize::MAX; n];
    let mut settled = vec![false; n];
    let mut touched = Vec::new();
    let mut total_settled = 0usize;
    let mut searches = 0usize;
    for (source_site, targets) in by_source {
        searches += 1;
        let source_cell = sites[source_site].cell;
        let mut remaining: BTreeMap<usize, usize> = targets
            .iter()
            .map(|&(site, pair_index)| (sites[site].cell, pair_index))
            .collect();
        let mut queue = BinaryHeap::new();
        distance[source_cell] = 0.0;
        predecessor[source_cell] = source_cell;
        touched.push(source_cell);
        queue.push(QueueEntry {
            cost: 0.0,
            cell: source_cell,
        });
        while let Some(QueueEntry { cost, cell }) = queue.pop() {
            if cost > distance[cell] || settled[cell] {
                continue;
            }
            settled[cell] = true;
            total_settled += 1;
            if total_settled > maximum_visits {
                return Err("route searches exceeded configured settled-cell budget");
            }
            if let Some(pair_index) = remaining.remove(&cell) {
                paths[pair_index] = Some(reconstruct_path(source_cell, cell, &predecessor)?);
                if remaining.is_empty() {
                    break;
                }
            }
            for &neighbor in tessellation.neighbors(cell) {
                if submerged[neighbor] || settled[neighbor] {
                    continue;
                }
                let edge = edge_metrics(tessellation, elevation, cell, neighbor, traversal)?;
                let next = cost + edge.symmetric_cost_km;
                if !next.is_finite() {
                    return Err("route search accumulated a non-finite generalized cost");
                }
                if next < distance[neighbor] {
                    if distance[neighbor].is_infinite() {
                        touched.push(neighbor);
                    }
                    distance[neighbor] = next;
                    predecessor[neighbor] = cell;
                    queue.push(QueueEntry {
                        cost: next,
                        cell: neighbor,
                    });
                }
            }
        }
        if !remaining.is_empty() {
            return Err("route target is unreachable within its terrestrial component");
        }
        for cell in touched.drain(..) {
            distance[cell] = f32::INFINITY;
            predecessor[cell] = usize::MAX;
            settled[cell] = false;
        }
    }
    let paths = paths
        .into_iter()
        .map(|path| path.ok_or("route candidate path was not searched"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((paths, searches, total_settled))
}

fn reconstruct_path(
    source: usize,
    target: usize,
    predecessor: &[usize],
) -> Result<Vec<usize>, &'static str> {
    let mut reverse = vec![target];
    let mut cell = target;
    while cell != source {
        cell = *predecessor
            .get(cell)
            .ok_or("route predecessor is out of range")?;
        if cell == usize::MAX || reverse.len() > predecessor.len() {
            return Err("route predecessor chain is incomplete or cyclic");
        }
        reverse.push(cell);
    }
    reverse.reverse();
    Ok(reverse)
}

#[allow(clippy::too_many_arguments)]
fn route_metrics(
    id: usize,
    from_site_id: usize,
    to_site_id: usize,
    landmass_anchor_cell: usize,
    ordered_cells: Vec<usize>,
    tessellation: &Tessellation,
    elevation: &[f32],
    repaired: &[bool],
    traversal: TraversalConfig,
) -> Result<AggregateRouteCandidate, &'static str> {
    if ordered_cells.len() < 2 {
        return Err("route path must contain at least one edge");
    }
    let mut physical = 0.0;
    let mut forward = 0.0;
    let mut reverse = 0.0;
    let mut symmetric = 0.0;
    let mut ascent = 0.0;
    let mut descent = 0.0;
    let mut maximum_grade = 0.0f32;
    let mut repaired_edges = 0usize;
    let mut repaired_length = 0.0;
    for pair in ordered_cells.windows(2) {
        if !tessellation.neighbors(pair[0]).contains(&pair[1]) {
            return Err("route path contains a non-adjacent cell pair");
        }
        let edge = edge_metrics(tessellation, elevation, pair[0], pair[1], traversal)?;
        physical += edge.distance_km;
        forward += edge.forward_cost_km;
        reverse += edge.reverse_cost_km;
        symmetric += edge.symmetric_cost_km;
        ascent += edge.ascent_km;
        descent += edge.descent_km;
        maximum_grade = maximum_grade.max(edge.absolute_grade);
        if repaired[pair[0]] || repaired[pair[1]] {
            repaired_edges += 1;
            repaired_length += edge.distance_km;
        }
    }
    let drainage_repaired_cell_count = ordered_cells.iter().filter(|&&cell| repaired[cell]).count();
    if [
        physical,
        forward,
        reverse,
        symmetric,
        ascent,
        descent,
        maximum_grade,
        repaired_length,
    ]
    .into_iter()
    .any(|value| !value.is_finite() || value < 0.0)
        || repaired_length > physical
    {
        return Err("route metrics must close to finite non-negative totals");
    }
    Ok(AggregateRouteCandidate {
        id,
        from_site_id,
        to_site_id,
        landmass_anchor_cell,
        ordered_cells,
        physical_length_km: physical,
        symmetric_generalized_cost_km: symmetric,
        forward_generalized_cost_km: forward,
        reverse_generalized_cost_km: reverse,
        ascent_km_from_from_site: ascent,
        descent_km_from_from_site: descent,
        maximum_absolute_grade: maximum_grade,
        drainage_repaired_cell_count,
        drainage_repaired_edge_count: repaired_edges,
        drainage_repaired_length_km: repaired_length,
        drainage_repaired_length_fraction: if physical > 0.0 {
            (repaired_length / physical).clamp(0.0, 1.0)
        } else {
            0.0
        },
        selection_role: None,
        network_detour_before_generalized_km: None,
        detour_ratio_before: None,
    })
}

fn edge_metrics(
    tessellation: &Tessellation,
    elevation: &[f32],
    from: usize,
    to: usize,
    traversal: TraversalConfig,
) -> Result<EdgeMetrics, &'static str> {
    let distance_km = physical_distance_km(tessellation, from, to);
    if !distance_km.is_finite() || distance_km <= 0.0 {
        return Err("route edge must have positive finite physical length");
    }
    let change_km = elevation_to_km(elevation[to] - elevation[from]);
    if !change_km.is_finite() {
        return Err("route elevation change must be finite");
    }
    let ascent = change_km.max(0.0);
    let descent = (-change_km).max(0.0);
    let forward =
        distance_km + traversal.uphill_penalty() * ascent + traversal.downhill_penalty() * descent;
    let reverse =
        distance_km + traversal.uphill_penalty() * descent + traversal.downhill_penalty() * ascent;
    let symmetric = 0.5 * (forward + reverse);
    let absolute_grade = change_km.abs() / distance_km;
    if [forward, reverse, symmetric, absolute_grade]
        .into_iter()
        .any(|value| !value.is_finite())
        || forward < distance_km
        || reverse < distance_km
        || symmetric < distance_km
    {
        return Err("route edge costs and grade must be finite and physically bounded");
    }
    Ok(EdgeMetrics {
        distance_km,
        forward_cost_km: forward,
        reverse_cost_km: reverse,
        symmetric_cost_km: symmetric,
        ascent_km: ascent,
        descent_km: descent,
        absolute_grade,
    })
}

fn physical_distance_km(tessellation: &Tessellation, from: usize, to: usize) -> f32 {
    let chord = (tessellation.cell_center(from) - tessellation.cell_center(to)).length();
    2.0 * PLANET_RADIUS_KM * (0.5 * chord).clamp(0.0, 1.0).asin()
}

#[derive(Clone, Copy, Debug)]
struct DivergenceIndices {
    physical: [usize; 2],
    distance_null: [usize; 2],
}

#[derive(Debug)]
struct QualifyingCorridor {
    indices: DivergenceIndices,
    physical: RouteSegmentTerrainEvidence,
    distance_null: RouteSegmentTerrainEvidence,
    generalized_saved: f32,
    elevation_saved: f32,
}

#[allow(clippy::too_many_arguments)]
fn assess_route_lower_corridor_from_fields(
    tessellation: &Tessellation,
    elevation: &[f32],
    is_submerged: &impl Fn(usize) -> bool,
    is_repaired: &impl Fn(usize) -> bool,
    physical: &AggregateRouteNetwork,
    distance_null: &AggregateRouteNetwork,
    candidate_route_id: usize,
) -> Result<RouteLowerCorridorAssessment, &'static str> {
    let n = tessellation.num_cells();
    if elevation.len() != n {
        return Err("route-corridor terrain must match tessellation");
    }
    if elevation.iter().any(|value| !value.is_finite()) {
        return Err("route-corridor terrain must be finite");
    }
    if physical.site_anchor_cells != distance_null.site_anchor_cells
        || physical.candidate_routes.len() != distance_null.candidate_routes.len()
    {
        return Err("route-corridor assessment requires matched networks");
    }
    if distance_null.traversal.uphill_penalty() != 0.0
        || distance_null.traversal.downhill_penalty() != 0.0
    {
        return Err("route-corridor comparison network must be a zero-grade distance null");
    }
    for (index, (physical_route, null_route)) in physical
        .candidate_routes
        .iter()
        .zip(&distance_null.candidate_routes)
        .enumerate()
    {
        if physical_route.id != index
            || null_route.id != index
            || (physical_route.from_site_id, physical_route.to_site_id)
                != (null_route.from_site_id, null_route.to_site_id)
        {
            return Err("route-corridor candidate identities must match");
        }
    }
    let physical_route = physical
        .candidate_routes
        .get(candidate_route_id)
        .ok_or("route-corridor candidate ID is out of range")?;
    let null_route = &distance_null.candidate_routes[candidate_route_id];
    validate_explanation_path(
        tessellation,
        is_submerged,
        &physical.site_anchor_cells,
        physical_route,
    )?;
    validate_explanation_path(
        tessellation,
        is_submerged,
        &distance_null.site_anchor_cells,
        null_route,
    )?;

    let omission = |reason,
                    divergent_segment_count,
                    distance_terrain_tradeoff_count,
                    generalized_cost_advantage_count,
                    lower_crossing_count| {
        RouteLowerCorridorAssessment::Omitted {
            omission: RouteLowerCorridorOmission {
                candidate_route_id,
                endpoint_site_ids: [physical_route.from_site_id, physical_route.to_site_id],
                physical_selection_role: physical_route.selection_role,
                distance_null_selection_role: null_route.selection_role,
                reason,
                divergent_segment_count,
                distance_terrain_tradeoff_count,
                generalized_cost_advantage_count,
                lower_crossing_count,
            },
        }
    };
    if physical.traversal.uphill_penalty() == 0.0 && physical.traversal.downhill_penalty() == 0.0 {
        return Ok(omission(
            RouteLowerCorridorOmissionReason::NoTerrainBurden,
            0,
            0,
            0,
            0,
        ));
    }
    if physical_route.ordered_cells == null_route.ordered_cells {
        return Ok(omission(
            RouteLowerCorridorOmissionReason::ExactPath,
            0,
            0,
            0,
            0,
        ));
    }

    let divergences =
        match elementary_divergences(&physical_route.ordered_cells, &null_route.ordered_cells) {
            Ok(divergences) => divergences,
            Err(RouteLowerCorridorOmissionReason::NonMonotonicCommonCellOrder) => {
                return Ok(omission(
                    RouteLowerCorridorOmissionReason::NonMonotonicCommonCellOrder,
                    0,
                    0,
                    0,
                    0,
                ));
            }
            Err(_) => {
                return Ok(omission(
                    RouteLowerCorridorOmissionReason::NoSimpleDivergence,
                    0,
                    0,
                    0,
                    0,
                ));
            }
        };
    if divergences.is_empty() {
        return Ok(omission(
            RouteLowerCorridorOmissionReason::NoSimpleDivergence,
            0,
            0,
            0,
            0,
        ));
    }

    let mut tradeoff_count = 0usize;
    let mut cost_advantage_count = 0usize;
    let mut lower_crossing_count = 0usize;
    let mut best: Option<QualifyingCorridor> = None;
    for indices in divergences.iter().copied() {
        let physical_cells =
            &physical_route.ordered_cells[indices.physical[0]..=indices.physical[1]];
        let null_cells =
            &null_route.ordered_cells[indices.distance_null[0]..=indices.distance_null[1]];
        let physical_metrics = segment_terrain_evidence(
            tessellation,
            elevation,
            is_repaired,
            physical_cells,
            physical.traversal,
        )?;
        let null_metrics = segment_terrain_evidence(
            tessellation,
            elevation,
            is_repaired,
            null_cells,
            physical.traversal,
        )?;
        if !roundoff_greater(
            physical_metrics.physical_length_km,
            null_metrics.physical_length_km,
        ) {
            continue;
        }
        tradeoff_count += 1;
        if !roundoff_greater(
            null_metrics.symmetric_generalized_cost_km,
            physical_metrics.symmetric_generalized_cost_km,
        ) {
            continue;
        }
        cost_advantage_count += 1;
        if !roundoff_greater(
            null_metrics.maximum_elevation_km,
            physical_metrics.maximum_elevation_km,
        ) {
            continue;
        }
        lower_crossing_count += 1;
        let generalized_saved = null_metrics.symmetric_generalized_cost_km
            - physical_metrics.symmetric_generalized_cost_km;
        let elevation_saved =
            null_metrics.maximum_elevation_km - physical_metrics.maximum_elevation_km;
        let replace = best.as_ref().is_none_or(|current| {
            generalized_saved
                .total_cmp(&current.generalized_saved)
                .then_with(|| elevation_saved.total_cmp(&current.elevation_saved))
                .then_with(|| {
                    let endpoints = [physical_cells[0], *physical_cells.last().unwrap()];
                    let current_cells = &physical_route.ordered_cells
                        [current.indices.physical[0]..=current.indices.physical[1]];
                    let current_endpoints = [current_cells[0], *current_cells.last().unwrap()];
                    current_endpoints.cmp(&endpoints)
                })
                .is_gt()
        });
        if replace {
            best = Some(QualifyingCorridor {
                indices,
                physical: physical_metrics,
                distance_null: null_metrics,
                generalized_saved,
                elevation_saved,
            });
        }
    }

    let Some(best) = best else {
        let reason = if tradeoff_count == 0 {
            RouteLowerCorridorOmissionReason::NoDistanceTerrainTradeoff
        } else if cost_advantage_count == 0 {
            RouteLowerCorridorOmissionReason::NoGeneralizedCostAdvantage
        } else {
            RouteLowerCorridorOmissionReason::NoLowerCrossing
        };
        return Ok(omission(
            reason,
            divergences.len(),
            tradeoff_count,
            cost_advantage_count,
            lower_crossing_count,
        ));
    };
    let physical_cells =
        &physical_route.ordered_cells[best.indices.physical[0]..=best.indices.physical[1]];
    let null_cells =
        &null_route.ordered_cells[best.indices.distance_null[0]..=best.indices.distance_null[1]];
    let (maximum_separation, separation_cells) =
        symmetric_path_separation(tessellation, physical_cells, null_cells);
    let endpoints = [physical_cells[0], *physical_cells.last().unwrap()];
    let whole_route_divergence = best.indices.physical
        == [0, physical_route.ordered_cells.len() - 1]
        && best.indices.distance_null == [0, null_route.ordered_cells.len() - 1];
    let physical_length_premium =
        best.physical.physical_length_km - best.distance_null.physical_length_km;
    let ascent_saved = best.distance_null.ascent_km - best.physical.ascent_km;
    let saved_fraction = best.generalized_saved / best.distance_null.symmetric_generalized_cost_km;
    Ok(RouteLowerCorridorAssessment::Explained {
        evidence: Box::new(RouteLowerCorridorEvidence {
            candidate_route_id,
            endpoint_site_ids: [physical_route.from_site_id, physical_route.to_site_id],
            physical_selection_role: physical_route.selection_role,
            distance_null_selection_role: null_route.selection_role,
            divergence_endpoint_cells: endpoints,
            physical_segment_cell_indices: best.indices.physical,
            distance_null_segment_cell_indices: best.indices.distance_null,
            physical_segment: best.physical,
            distance_null_segment: best.distance_null,
            physical_length_premium_km: physical_length_premium,
            generalized_cost_saved_km: best.generalized_saved,
            generalized_cost_saved_fraction: saved_fraction,
            ascent_saved_km: ascent_saved,
            maximum_elevation_saved_km: best.elevation_saved,
            divergence_endpoint_distance_km: physical_distance_km(
                tessellation,
                endpoints[0],
                endpoints[1],
            ),
            maximum_symmetric_cell_center_separation_km: maximum_separation,
            maximum_symmetric_cell_center_separation_cells: separation_cells,
            whole_route_divergence,
            interpretation: "lower-terrain-corridor-relative-to-distance-null",
            limitation: "route and scalar terrain evidence do not establish a geomorphic gap, pass, ridge crossing, or chokepoint",
        }),
    })
}

fn validate_explanation_path(
    tessellation: &Tessellation,
    is_submerged: &impl Fn(usize) -> bool,
    site_anchor_cells: &[usize],
    route: &AggregateRouteCandidate,
) -> Result<(), &'static str> {
    let n = tessellation.num_cells();
    if route.from_site_id >= site_anchor_cells.len()
        || route.to_site_id >= site_anchor_cells.len()
        || route.ordered_cells.len() < 2
        || route.ordered_cells.first() != Some(&site_anchor_cells[route.from_site_id])
        || route.ordered_cells.last() != Some(&site_anchor_cells[route.to_site_id])
    {
        return Err("route-corridor path must match its canonical site anchors");
    }
    let mut seen = BTreeSet::new();
    for &cell in &route.ordered_cells {
        if cell >= n || is_submerged(cell) {
            return Err("route-corridor path must stay on in-range land cells");
        }
        if !seen.insert(cell) {
            return Err("route-corridor path must be simple");
        }
    }
    if route
        .ordered_cells
        .windows(2)
        .any(|pair| !tessellation.neighbors(pair[0]).contains(&pair[1]))
    {
        return Err("route-corridor path must follow Voronoi adjacency");
    }
    Ok(())
}

fn elementary_divergences(
    physical: &[usize],
    distance_null: &[usize],
) -> Result<Vec<DivergenceIndices>, RouteLowerCorridorOmissionReason> {
    let mut null_index = BTreeMap::new();
    for (index, &cell) in distance_null.iter().enumerate() {
        null_index.insert(cell, index);
    }
    let mut common = Vec::new();
    for (physical_index, &cell) in physical.iter().enumerate() {
        if let Some(&distance_null_index) = null_index.get(&cell) {
            common.push((physical_index, distance_null_index));
        }
    }
    if common.windows(2).any(|pair| pair[0].1 >= pair[1].1) {
        return Err(RouteLowerCorridorOmissionReason::NonMonotonicCommonCellOrder);
    }
    let mut result = Vec::new();
    for pair in common.windows(2) {
        let physical_slice = &physical[pair[0].0..=pair[1].0];
        let null_slice = &distance_null[pair[0].1..=pair[1].1];
        if physical_slice != null_slice {
            result.push(DivergenceIndices {
                physical: [pair[0].0, pair[1].0],
                distance_null: [pair[0].1, pair[1].1],
            });
        }
    }
    Ok(result)
}

fn segment_terrain_evidence(
    tessellation: &Tessellation,
    elevation: &[f32],
    is_repaired: &impl Fn(usize) -> bool,
    cells: &[usize],
    traversal: TraversalConfig,
) -> Result<RouteSegmentTerrainEvidence, &'static str> {
    if cells.len() < 2 {
        return Err("route-corridor segment must contain an edge");
    }
    let mut physical_length = 0.0;
    let mut generalized_cost = 0.0;
    let mut ascent = 0.0;
    let mut descent = 0.0;
    let mut maximum_grade = 0.0f32;
    let mut repaired_edges = 0usize;
    let mut repaired_length = 0.0;
    for pair in cells.windows(2) {
        let edge = edge_metrics(tessellation, elevation, pair[0], pair[1], traversal)?;
        physical_length += edge.distance_km;
        generalized_cost += edge.symmetric_cost_km;
        ascent += edge.ascent_km;
        descent += edge.descent_km;
        maximum_grade = maximum_grade.max(edge.absolute_grade);
        if is_repaired(pair[0]) || is_repaired(pair[1]) {
            repaired_edges += 1;
            repaired_length += edge.distance_km;
        }
    }
    let mut maximum_elevation_cell = cells[0];
    for &cell in cells.iter().skip(1) {
        if elevation[cell] > elevation[maximum_elevation_cell]
            || (elevation[cell].to_bits() == elevation[maximum_elevation_cell].to_bits()
                && cell < maximum_elevation_cell)
        {
            maximum_elevation_cell = cell;
        }
    }
    let maximum_elevation_km = elevation_to_km(elevation[maximum_elevation_cell]);
    if [
        physical_length,
        generalized_cost,
        ascent,
        descent,
        maximum_grade,
        repaired_length,
        maximum_elevation_km,
    ]
    .into_iter()
    .any(|value| !value.is_finite() || value < 0.0)
        || repaired_length > physical_length
    {
        return Err("route-corridor segment metrics must close to finite non-negative totals");
    }
    Ok(RouteSegmentTerrainEvidence {
        edge_count: cells.len() - 1,
        physical_length_km: physical_length,
        symmetric_generalized_cost_km: generalized_cost,
        ascent_km: ascent,
        descent_km: descent,
        maximum_elevation_km,
        maximum_elevation_cell,
        maximum_absolute_grade: maximum_grade,
        drainage_repaired_cell_count: cells.iter().filter(|&&cell| is_repaired(cell)).count(),
        drainage_repaired_edge_count: repaired_edges,
        drainage_repaired_length_km: repaired_length,
        drainage_repaired_length_fraction: (repaired_length / physical_length).clamp(0.0, 1.0),
    })
}

fn roundoff_greater(a: f32, b: f32) -> bool {
    let guard = 16.0 * f32::EPSILON * a.abs().max(b.abs()).max(1.0);
    a - b > guard
}

fn symmetric_path_separation(
    tessellation: &Tessellation,
    physical: &[usize],
    distance_null: &[usize],
) -> (f32, [usize; 2]) {
    fn directed(
        tessellation: &Tessellation,
        from: &[usize],
        to: &[usize],
        reverse_pair: bool,
    ) -> (f32, [usize; 2]) {
        let mut maximum = (-1.0f32, [usize::MAX; 2]);
        for &from_cell in from {
            let mut nearest = (f32::INFINITY, usize::MAX);
            for &to_cell in to {
                let distance = physical_distance_km(tessellation, from_cell, to_cell);
                if distance < nearest.0
                    || (distance.to_bits() == nearest.0.to_bits() && to_cell < nearest.1)
                {
                    nearest = (distance, to_cell);
                }
            }
            let cells = if reverse_pair {
                [nearest.1, from_cell]
            } else {
                [from_cell, nearest.1]
            };
            if nearest.0 > maximum.0
                || (nearest.0.to_bits() == maximum.0.to_bits() && cells < maximum.1)
            {
                maximum = (nearest.0, cells);
            }
        }
        maximum
    }
    let forward = directed(tessellation, physical, distance_null, false);
    let reverse = directed(tessellation, distance_null, physical, true);
    if reverse.0 > forward.0
        || (reverse.0.to_bits() == forward.0.to_bits() && reverse.1 < forward.1)
    {
        reverse
    } else {
        forward
    }
}

fn selected_network_distance(
    site_count: usize,
    routes: &[AggregateRouteCandidate],
    selected: &[usize],
    source: usize,
    target: usize,
) -> Option<f32> {
    let mut distance = vec![f32::INFINITY; site_count];
    let mut queue = BinaryHeap::new();
    distance[source] = 0.0;
    queue.push(QueueEntry {
        cost: 0.0,
        cell: source,
    });
    while let Some(QueueEntry { cost, cell }) = queue.pop() {
        if cost > distance[cell] {
            continue;
        }
        if cell == target {
            return Some(cost);
        }
        for &route_id in selected {
            let route = &routes[route_id];
            let neighbor = if route.from_site_id == cell {
                Some(route.to_site_id)
            } else if route.to_site_id == cell {
                Some(route.from_site_id)
            } else {
                None
            };
            if let Some(neighbor) = neighbor {
                let next = cost + route.symmetric_generalized_cost_km;
                if next < distance[neighbor] {
                    distance[neighbor] = next;
                    queue.push(QueueEntry {
                        cost: next,
                        cell: neighbor,
                    });
                }
            }
        }
    }
    None
}

fn validate_selected_components(
    site_count: usize,
    components: &[TerrestrialRouteComponent],
    routes: &[AggregateRouteCandidate],
) -> Result<(), &'static str> {
    for component in components
        .iter()
        .filter(|component| component.site_ids.len() > 1)
    {
        let mut dsu = DisjointSet::new(site_count);
        for &route_id in &component.selected_route_ids {
            let route = routes
                .get(route_id)
                .ok_or("selected route ID is out of range")?;
            dsu.union(route.from_site_id, route.to_site_id);
        }
        let root = dsu.find(component.site_ids[0]);
        if component
            .site_ids
            .iter()
            .any(|&site| dsu.find(site) != root)
        {
            return Err("selected routes do not connect an occupied landmass");
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.find(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            return false;
        }
        if self.rank[a] < self.rank[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        if self.rank[a] == self.rank[b] {
            self.rank[a] += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    fn tessellation(cells: usize) -> Tessellation {
        let mut rng = ChaCha8Rng::seed_from_u64(84_221);
        Tessellation::generate(cells, 0, &mut rng)
    }

    fn traversal() -> TraversalConfig {
        TraversalConfig::new(12.0, 3.0).unwrap()
    }

    fn route_config() -> RouteNetworkConfig {
        RouteNetworkConfig {
            nearest_neighbors_per_site: 2,
            maximum_candidate_pair_count: 64,
            maximum_total_search_cell_visits: 2_000_000,
            maximum_extra_links: 2,
            minimum_extra_link_detour_ratio: 1.3,
        }
    }

    #[test]
    fn sparse_route_forest_is_deterministic_connected_and_bounded() {
        let tess = tessellation(600);
        let n = tess.num_cells();
        let elevation = vec![0.0; n];
        let submerged = vec![false; n];
        let repaired = vec![false; n];
        let sites: Vec<SiteAnchor> = [0, 100, 200, 300, 400, 500]
            .into_iter()
            .enumerate()
            .map(|(id, cell)| SiteAnchor { id, cell })
            .collect();
        let first = build_route_network_from_fields(
            &tess,
            &elevation,
            &submerged,
            &repaired,
            &sites,
            traversal(),
            route_config().validate().unwrap(),
        )
        .unwrap();
        let second = build_route_network_from_fields(
            &tess,
            &elevation,
            &submerged,
            &repaired,
            &sites,
            traversal(),
            route_config().validate().unwrap(),
        )
        .unwrap();

        assert_eq!(first.occupied_landmass_count, 1);
        assert_eq!(first.routable_landmass_count, 1);
        assert_eq!(first.backbone_route_count, sites.len() - 1);
        assert!(first.extra_route_count <= route_config().maximum_extra_links);
        assert!(first.candidate_pair_count < sites.len() * (sites.len() - 1) / 2);
        assert!(first.route_search_count <= sites.len());
        assert!(first.search_settled_cell_count <= route_config().maximum_total_search_cell_visits);
        assert_eq!(first.selected_route_ids, second.selected_route_ids);
        assert_eq!(first.candidate_routes.len(), second.candidate_routes.len());
        for (a, b) in first.candidate_routes.iter().zip(&second.candidate_routes) {
            assert_eq!(
                (a.from_site_id, a.to_site_id),
                (b.from_site_id, b.to_site_id)
            );
            assert_eq!(a.ordered_cells, b.ordered_cells);
            assert_eq!(a.selection_role, b.selection_role);
            assert!(a.physical_length_km > 0.0);
            assert_eq!(
                a.physical_length_km.to_bits(),
                a.symmetric_generalized_cost_km.to_bits()
            );
            assert!(a
                .ordered_cells
                .windows(2)
                .all(|pair| tess.neighbors(pair[0]).contains(&pair[1])));
        }
    }

    #[test]
    fn low_gap_redirects_physical_route_but_zero_grade_does_not_claim_it() {
        let tess = tessellation(900);
        let n = tess.num_cells();
        let source = (0..n)
            .min_by(|&a, &b| tess.cell_center(a).x.total_cmp(&tess.cell_center(b).x))
            .unwrap();
        let target = (0..n)
            .max_by(|&a, &b| tess.cell_center(a).x.total_cmp(&tess.cell_center(b).x))
            .unwrap();
        let sites = [
            SiteAnchor {
                id: 0,
                cell: source,
            },
            SiteAnchor {
                id: 1,
                cell: target,
            },
        ];
        let submerged = vec![false; n];
        let repaired = vec![false; n];
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
        let mut config = route_config();
        config.nearest_neighbors_per_site = 1;
        config.maximum_candidate_pair_count = 1;
        config.maximum_extra_links = 0;
        let closed_route = build_route_network_from_fields(
            &tess,
            &closed,
            &submerged,
            &repaired,
            &sites,
            traversal(),
            config,
        )
        .unwrap();
        let open_route = build_route_network_from_fields(
            &tess,
            &open,
            &submerged,
            &repaired,
            &sites,
            traversal(),
            config,
        )
        .unwrap();
        assert!(
            open_route.candidate_routes[0].symmetric_generalized_cost_km
                < closed_route.candidate_routes[0].symmetric_generalized_cost_km
        );
        assert!(open_route.candidate_routes[0]
            .ordered_cells
            .iter()
            .any(|&cell| {
                let center = tess.cell_center(cell);
                center.x.abs() < 0.28 && center.z > 0.72
            }));

        let flat = TraversalConfig::new(0.0, 0.0).unwrap();
        let flat_closed = build_route_network_from_fields(
            &tess, &closed, &submerged, &repaired, &sites, flat, config,
        )
        .unwrap();
        let flat_open = build_route_network_from_fields(
            &tess, &open, &submerged, &repaired, &sites, flat, config,
        )
        .unwrap();
        assert_eq!(
            flat_closed.candidate_routes[0].ordered_cells,
            flat_open.candidate_routes[0].ordered_cells
        );
        assert_eq!(
            flat_closed.candidate_routes[0]
                .symmetric_generalized_cost_km
                .to_bits(),
            flat_open.candidate_routes[0]
                .symmetric_generalized_cost_km
                .to_bits()
        );

        let assessment = assess_route_lower_corridor_from_fields(
            &tess,
            &open,
            &|cell| submerged[cell],
            &|cell| repaired[cell],
            &open_route,
            &flat_open,
            0,
        )
        .unwrap();
        let RouteLowerCorridorAssessment::Explained { evidence } = assessment else {
            panic!("the manufactured opening should support a lower-corridor explanation");
        };
        assert_eq!(evidence.endpoint_site_ids, [0, 1]);
        assert_eq!(
            evidence.physical_selection_role,
            Some(RouteSelectionRole::Backbone)
        );
        assert_eq!(
            evidence.distance_null_selection_role,
            Some(RouteSelectionRole::Backbone)
        );
        assert!(evidence.physical_length_premium_km > 0.0);
        assert!(evidence.generalized_cost_saved_km > 0.0);
        assert!(evidence.maximum_elevation_saved_km > 0.0);
        assert!(evidence.ascent_saved_km > 0.0);
        assert!(evidence.divergence_endpoint_distance_km > 0.0);
        assert!(evidence.maximum_symmetric_cell_center_separation_km > 0.0);
        assert!(
            open[evidence.physical_segment.maximum_elevation_cell]
                < open[evidence.distance_null_segment.maximum_elevation_cell]
        );
        assert!(
            evidence.physical_segment.symmetric_generalized_cost_km
                < evidence.distance_null_segment.symmetric_generalized_cost_km
        );
        assert!(
            evidence.distance_null_segment.symmetric_generalized_cost_km
                > evidence.distance_null_segment.physical_length_km
        );
        assert_eq!(
            evidence.interpretation,
            "lower-terrain-corridor-relative-to-distance-null"
        );
    }

    #[test]
    fn flat_matched_paths_omit_corridor_explanation() {
        let tess = tessellation(300);
        let n = tess.num_cells();
        let sites = [
            SiteAnchor { id: 0, cell: 0 },
            SiteAnchor { id: 1, cell: n / 2 },
        ];
        let elevation = vec![0.0; n];
        let submerged = vec![false; n];
        let repaired = vec![false; n];
        let mut config = route_config();
        config.nearest_neighbors_per_site = 1;
        config.maximum_candidate_pair_count = 1;
        config.maximum_extra_links = 0;
        let physical = build_route_network_from_fields(
            &tess,
            &elevation,
            &submerged,
            &repaired,
            &sites,
            traversal(),
            config,
        )
        .unwrap();
        let distance_null = build_route_network_from_fields(
            &tess,
            &elevation,
            &submerged,
            &repaired,
            &sites,
            TraversalConfig::new(0.0, 0.0).unwrap(),
            config,
        )
        .unwrap();
        let assessment = assess_route_lower_corridor_from_fields(
            &tess,
            &elevation,
            &|cell| submerged[cell],
            &|cell| repaired[cell],
            &physical,
            &distance_null,
            0,
        )
        .unwrap();
        let RouteLowerCorridorAssessment::Omitted { omission } = assessment else {
            panic!("identical flat paths must not acquire a terrain explanation");
        };
        assert_eq!(omission.reason, RouteLowerCorridorOmissionReason::ExactPath);
        assert_eq!(omission.divergent_segment_count, 0);
    }

    #[test]
    fn crossed_common_cell_order_is_omitted_without_inventing_a_bubble() {
        let tess = tessellation(300);
        let n = tess.num_cells();
        let (a, b, source, target) = (0..n)
            .find_map(|a| {
                tess.neighbors(a).iter().copied().find_map(|b| {
                    if b <= a {
                        return None;
                    }
                    let common: Vec<usize> = tess
                        .neighbors(a)
                        .iter()
                        .copied()
                        .filter(|cell| tess.neighbors(b).contains(cell))
                        .collect();
                    (common.len() >= 2).then(|| (a, b, common[0], common[1]))
                })
            })
            .expect("spherical Delaunay edge should have two triangle neighbors");
        let sites = [
            SiteAnchor {
                id: 0,
                cell: source,
            },
            SiteAnchor {
                id: 1,
                cell: target,
            },
        ];
        let elevation = vec![0.0; n];
        let submerged = vec![false; n];
        let repaired = vec![false; n];
        let mut config = route_config();
        config.nearest_neighbors_per_site = 1;
        config.maximum_candidate_pair_count = 1;
        config.maximum_extra_links = 0;
        let mut physical = build_route_network_from_fields(
            &tess,
            &elevation,
            &submerged,
            &repaired,
            &sites,
            traversal(),
            config,
        )
        .unwrap();
        let mut distance_null = build_route_network_from_fields(
            &tess,
            &elevation,
            &submerged,
            &repaired,
            &sites,
            TraversalConfig::new(0.0, 0.0).unwrap(),
            config,
        )
        .unwrap();
        physical.candidate_routes[0].ordered_cells = vec![source, a, b, target];
        distance_null.candidate_routes[0].ordered_cells = vec![source, b, a, target];
        let assessment = assess_route_lower_corridor_from_fields(
            &tess,
            &elevation,
            &|cell| submerged[cell],
            &|cell| repaired[cell],
            &physical,
            &distance_null,
            0,
        )
        .unwrap();
        let RouteLowerCorridorAssessment::Omitted { omission } = assessment else {
            panic!("crossed common-cell order must not be converted into a bubble");
        };
        assert_eq!(
            omission.reason,
            RouteLowerCorridorOmissionReason::NonMonotonicCommonCellOrder
        );
    }

    #[test]
    fn corridor_assessment_rejects_non_simple_paths() {
        let tess = tessellation(200);
        let n = tess.num_cells();
        let sites = [
            SiteAnchor { id: 0, cell: 0 },
            SiteAnchor { id: 1, cell: n / 2 },
        ];
        let elevation = vec![0.0; n];
        let submerged = vec![false; n];
        let repaired = vec![false; n];
        let mut config = route_config();
        config.nearest_neighbors_per_site = 1;
        config.maximum_candidate_pair_count = 1;
        config.maximum_extra_links = 0;
        let physical = build_route_network_from_fields(
            &tess,
            &elevation,
            &submerged,
            &repaired,
            &sites,
            traversal(),
            config,
        )
        .unwrap();
        let mut distance_null = build_route_network_from_fields(
            &tess,
            &elevation,
            &submerged,
            &repaired,
            &sites,
            TraversalConfig::new(0.0, 0.0).unwrap(),
            config,
        )
        .unwrap();
        let repeated = distance_null.candidate_routes[0].ordered_cells[0];
        distance_null.candidate_routes[0]
            .ordered_cells
            .insert(1, repeated);
        assert!(assess_route_lower_corridor_from_fields(
            &tess,
            &elevation,
            &|cell| submerged[cell],
            &|cell| repaired[cell],
            &physical,
            &distance_null,
            0,
        )
        .is_err());
    }

    #[test]
    fn corridor_segment_repair_evidence_closes() {
        let tess = tessellation(80);
        let a = 0;
        let b = tess.neighbors(a)[0];
        let c = tess
            .neighbors(b)
            .iter()
            .copied()
            .find(|&cell| cell != a)
            .unwrap();
        let elevation = vec![0.0; tess.num_cells()];
        let mut repaired = vec![false; tess.num_cells()];
        repaired[b] = true;
        let evidence = segment_terrain_evidence(
            &tess,
            &elevation,
            &|cell| repaired[cell],
            &[a, b, c],
            traversal(),
        )
        .unwrap();
        assert_eq!(evidence.edge_count, 2);
        assert_eq!(evidence.drainage_repaired_cell_count, 1);
        assert_eq!(evidence.drainage_repaired_edge_count, 2);
        assert_eq!(
            evidence.drainage_repaired_length_km.to_bits(),
            evidence.physical_length_km.to_bits()
        );
        assert_eq!(evidence.drainage_repaired_length_fraction, 1.0);
    }

    #[test]
    fn disconnected_land_groups_form_a_forest_with_an_isolated_site() {
        let tess = tessellation(900);
        let n = tess.num_cells();
        let submerged: Vec<bool> = (0..n)
            .map(|cell| {
                let x = tess.cell_center(cell).x;
                !(-1.0..-0.25).contains(&x) && !(0.65..=1.0).contains(&x)
            })
            .collect();
        let mut left: Vec<usize> = (0..n)
            .filter(|&cell| tess.cell_center(cell).x < -0.7)
            .collect();
        left.sort_unstable();
        let right = (0..n).find(|&cell| tess.cell_center(cell).x > 0.8).unwrap();
        let sites = [
            SiteAnchor {
                id: 0,
                cell: left[0],
            },
            SiteAnchor {
                id: 1,
                cell: *left.last().unwrap(),
            },
            SiteAnchor { id: 2, cell: right },
        ];
        let result = build_route_network_from_fields(
            &tess,
            &vec![0.0; n],
            &submerged,
            &vec![false; n],
            &sites,
            traversal(),
            route_config(),
        )
        .unwrap();
        assert_eq!(result.occupied_landmass_count, 2);
        assert_eq!(result.routable_landmass_count, 1);
        assert_eq!(result.isolated_site_ids, vec![2]);
        assert_eq!(result.backbone_route_count, 1);
        assert!(result
            .candidate_routes
            .iter()
            .all(|route| { route.ordered_cells.iter().all(|&cell| !submerged[cell]) }));
    }

    #[test]
    fn route_metrics_close_ascent_reverse_and_repair_provenance() {
        let tess = tessellation(80);
        let a = 0;
        let b = tess.neighbors(a)[0];
        let mut elevation = vec![0.0; tess.num_cells()];
        elevation[b] = 0.2;
        let mut repaired = vec![false; tess.num_cells()];
        repaired[a] = true;
        let route = route_metrics(
            0,
            0,
            1,
            0,
            vec![a, b],
            &tess,
            &elevation,
            &repaired,
            traversal(),
        )
        .unwrap();
        assert!(route.ascent_km_from_from_site > 0.0);
        assert_eq!(route.descent_km_from_from_site, 0.0);
        assert!(route.forward_generalized_cost_km > route.reverse_generalized_cost_km);
        assert_eq!(route.drainage_repaired_cell_count, 1);
        assert_eq!(route.drainage_repaired_edge_count, 1);
        assert_eq!(route.drainage_repaired_length_fraction, 1.0);
        assert_eq!(
            route.symmetric_generalized_cost_km.to_bits(),
            (0.5 * (route.forward_generalized_cost_km + route.reverse_generalized_cost_km))
                .to_bits()
        );
    }

    #[test]
    fn route_config_and_budgets_are_rejected_explicitly() {
        let mut invalid = route_config();
        invalid.nearest_neighbors_per_site = 0;
        assert!(invalid.validate().is_err());
        invalid = route_config();
        invalid.maximum_candidate_pair_count = 129;
        assert!(invalid.validate().is_err());
        invalid = route_config();
        invalid.maximum_total_search_cell_visits = 0;
        assert!(invalid.validate().is_err());
        invalid = route_config();
        invalid.maximum_extra_links = 9;
        assert!(invalid.validate().is_err());
        invalid = route_config();
        invalid.minimum_extra_link_detour_ratio = f32::NAN;
        assert!(invalid.validate().is_err());

        let tess = tessellation(100);
        let sites = [SiteAnchor { id: 0, cell: 0 }, SiteAnchor { id: 1, cell: 1 }];
        let mut budget = route_config();
        budget.maximum_candidate_pair_count = 1;
        budget.maximum_total_search_cell_visits = 1;
        assert!(build_route_network_from_fields(
            &tess,
            &vec![0.0; tess.num_cells()],
            &vec![false; tess.num_cells()],
            &vec![false; tess.num_cells()],
            &sites,
            traversal(),
            budget,
        )
        .is_err());
    }
}
