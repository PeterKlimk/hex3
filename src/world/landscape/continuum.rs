//! Unfilled finite-volume flow routing for a cell-mean surface.
//!
//! This is an isolated C0 research operator. Its direct controls partition
//! water only across strictly downhill physical faces. A separate derived
//! route fills depressions from genuine portals and resolves equal-level flow
//! with an independently computed graph potential; neither path mutates the
//! physical surface. Unresolved sinks are reported as instantaneous storage.

use super::{BoundaryFaceCondition, LandscapeMesh, OutletPortalId};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BinaryHeap, collections::VecDeque, fmt};

/// Rule used to partition a cell's water among its downhill faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowPartition {
    /// Weight each downhill face by `face_width * physical_slope` (p = 1).
    MfdSlope,
}

/// Directed face fluxes and continuum specific-discharge diagnostics.
///
/// Face arrays use the mesh's directed CSR edge indexing. A flux is positive
/// from the owning cell to that edge's neighbor. For each reciprocal pair at
/// most one direction can be nonzero because routing is strictly downhill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceFlowCache {
    pub partition: FlowPartition,
    /// Fraction of the owning cell's available water sent over each edge.
    pub directed_edge_fraction: Vec<f64>,
    /// Water flux sent over each directed edge (km^3/Myr).
    pub directed_edge_flux_km3_myr: Vec<f64>,
    /// Fractions and outflows aligned with `LandscapeMesh::boundary_faces`.
    /// Closed faces and the closed-boundary analytic control remain zero.
    pub boundary_face_fraction: Vec<f64>,
    pub boundary_face_flux_km3_myr: Vec<f64>,
    /// Total outflow by stable semantic portal, in mesh portal order.
    pub portal_outflow_km3_myr: Vec<(OutletPortalId, f64)>,
    /// Derived surface used for routing. Unfilled controls copy physical elevation.
    pub routing_elevation_km: Vec<f64>,
    /// Graph distance toward an exit within equal-filled components.
    pub flat_potential: Vec<Option<u32>>,
    /// Local water supplied to each cell (km^3/Myr).
    pub local_supply_km3_myr: Vec<f64>,
    /// Local plus accumulated upstream water arriving at each cell.
    pub available_supply_km3_myr: Vec<f64>,
    /// Least-squares reconstruction from signed face samples Q / face width.
    pub specific_discharge_vector_km2_myr: Vec<DVec3>,
    pub specific_discharge_km2_myr: Vec<f64>,
    /// Water retained at cells having no strictly downhill neighbor.
    pub sink_storage_km3_myr: Vec<f64>,
    /// High-to-low topological order used by the acyclic routing pass.
    pub high_to_low_order: Vec<usize>,
    pub total_supply_km3_myr: f64,
    pub total_portal_outflow_km3_myr: f64,
    pub total_sink_storage_km3_myr: f64,
}

/// Portal-seeded fill and independently derived equal-level routing potential.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepressionRoutingSurface {
    pub filled_elevation_km: Vec<f64>,
    pub flat_potential: Vec<Option<u32>>,
}

impl FaceFlowCache {
    /// Route cell-local supply over the physical, unfilled elevation field.
    pub fn route(
        mesh: &LandscapeMesh,
        elevation_km: &[f64],
        local_supply_km3_myr: &[f64],
        partition: FlowPartition,
    ) -> Result<Self, ContinuumFlowError> {
        Self::route_impl(
            mesh,
            elevation_km,
            local_supply_km3_myr,
            partition,
            false,
            None,
        )
    }

    /// Route over internal faces and open physical portal subfaces.
    ///
    /// Each open-face weight shares the same normalization as internal faces:
    /// `face_width * max((z_cell - z_base) / center_distance, 0)`.
    pub fn route_with_portals(
        mesh: &LandscapeMesh,
        elevation_km: &[f64],
        local_supply_km3_myr: &[f64],
        partition: FlowPartition,
    ) -> Result<Self, ContinuumFlowError> {
        Self::route_impl(
            mesh,
            elevation_km,
            local_supply_km3_myr,
            partition,
            true,
            None,
        )
    }

    /// Route across a portal-seeded fill without modifying physical elevation.
    pub fn route_with_depressions(
        mesh: &LandscapeMesh,
        physical_elevation_km: &[f64],
        local_supply_km3_myr: &[f64],
        partition: FlowPartition,
    ) -> Result<Self, ContinuumFlowError> {
        let surface = DepressionRoutingSurface::derive(mesh, physical_elevation_km)?;
        Self::route_impl(
            mesh,
            physical_elevation_km,
            local_supply_km3_myr,
            partition,
            true,
            Some(&surface),
        )
    }

    fn route_impl(
        mesh: &LandscapeMesh,
        elevation_km: &[f64],
        local_supply_km3_myr: &[f64],
        partition: FlowPartition,
        use_portals: bool,
        depression_surface: Option<&DepressionRoutingSurface>,
    ) -> Result<Self, ContinuumFlowError> {
        mesh.validate()
            .map_err(|error| ContinuumFlowError(error.to_string()))?;
        let n = mesh.cell_count();
        if elevation_km.len() != n {
            return Err(ContinuumFlowError(format!(
                "elevation length {}, expected {n}",
                elevation_km.len()
            )));
        }
        if local_supply_km3_myr.len() != n {
            return Err(ContinuumFlowError(format!(
                "supply length {}, expected {n}",
                local_supply_km3_myr.len()
            )));
        }
        if elevation_km.iter().any(|z| !z.is_finite()) {
            return Err(ContinuumFlowError("elevation must be finite".into()));
        }
        if local_supply_km3_myr
            .iter()
            .any(|q| !q.is_finite() || *q < 0.0)
        {
            return Err(ContinuumFlowError(
                "local supply must be finite and non-negative".into(),
            ));
        }

        let routing_elevation_km = depression_surface.map_or_else(
            || elevation_km.to_vec(),
            |surface| surface.filled_elevation_km.clone(),
        );
        let flat_potential = depression_surface
            .map_or_else(|| vec![None; n], |surface| surface.flat_potential.clone());
        let mut high_to_low_order: Vec<_> = (0..n).collect();
        high_to_low_order.sort_unstable_by(|&a, &b| {
            routing_elevation_km[b]
                .total_cmp(&routing_elevation_km[a])
                .then_with(|| flat_potential[b].cmp(&flat_potential[a]))
                .then_with(|| a.cmp(&b))
        });

        let m = mesh.edge_neighbor.len();
        let mut directed_edge_fraction = vec![0.0; m];
        let mut directed_edge_flux_km3_myr = vec![0.0; m];
        let mut boundary_face_fraction = vec![0.0; mesh.boundary_faces.len()];
        let mut boundary_face_flux_km3_myr = vec![0.0; mesh.boundary_faces.len()];
        let mut boundary_faces_by_cell = vec![Vec::new(); n];
        if use_portals {
            for (face_index, face) in mesh.boundary_faces.iter().enumerate() {
                boundary_faces_by_cell[face.cell as usize].push(face_index);
            }
        }
        let mut available_supply_km3_myr = local_supply_km3_myr.to_vec();
        let mut sink_storage_km3_myr = vec![0.0; n];
        for &cell in &high_to_low_order {
            let start = mesh.edge_offsets[cell] as usize;
            let end = mesh.edge_offsets[cell + 1] as usize;
            let mut weight_sum = 0.0;
            for (edge, fraction_slot) in directed_edge_fraction
                .iter_mut()
                .enumerate()
                .take(end)
                .skip(start)
            {
                let neighbor = mesh.edge_neighbor[edge] as usize;
                let drop = routing_elevation_km[cell] - routing_elevation_km[neighbor];
                let weight = if drop > 0.0 {
                    let slope = drop / f64::from(mesh.edge_distance_km[edge]);
                    f64::from(mesh.edge_face_width_km[edge]) * slope
                } else if drop == 0.0
                    && matches!(
                        (flat_potential[cell], flat_potential[neighbor]),
                        (Some(from), Some(to)) if to < from
                    )
                {
                    f64::from(mesh.edge_face_width_km[edge])
                        / f64::from(mesh.edge_distance_km[edge])
                } else {
                    0.0
                };
                *fraction_slot = weight;
                weight_sum += weight;
            }
            for &face_index in &boundary_faces_by_cell[cell] {
                let face = &mesh.boundary_faces[face_index];
                let weight = match face.condition {
                    BoundaryFaceCondition::OpenBaseLevel {
                        elevation_km: base, ..
                    } => {
                        let drop = routing_elevation_km[cell] - f64::from(base);
                        if drop > 0.0 {
                            face.width_km * drop / face.center_distance_km
                        } else if drop == 0.0 && flat_potential[cell] == Some(0) {
                            face.width_km / face.center_distance_km
                        } else {
                            0.0
                        }
                    }
                    BoundaryFaceCondition::Closed => 0.0,
                };
                boundary_face_fraction[face_index] = weight;
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                for (edge, fraction_slot) in directed_edge_fraction
                    .iter_mut()
                    .enumerate()
                    .take(end)
                    .skip(start)
                {
                    let weight = *fraction_slot;
                    if weight == 0.0 {
                        continue;
                    }
                    let fraction = weight / weight_sum;
                    let flux = available_supply_km3_myr[cell] * fraction;
                    *fraction_slot = fraction;
                    directed_edge_flux_km3_myr[edge] = flux;
                    let neighbor = mesh.edge_neighbor[edge] as usize;
                    available_supply_km3_myr[neighbor] += flux;
                }
                for &face_index in &boundary_faces_by_cell[cell] {
                    let weight = boundary_face_fraction[face_index];
                    if weight == 0.0 {
                        continue;
                    }
                    let fraction = weight / weight_sum;
                    boundary_face_fraction[face_index] = fraction;
                    boundary_face_flux_km3_myr[face_index] =
                        available_supply_km3_myr[cell] * fraction;
                }
            } else {
                sink_storage_km3_myr[cell] = available_supply_km3_myr[cell];
            }
        }

        let reverse_edge = reciprocal_edges(mesh)?;
        let mut specific_discharge_vector_km2_myr = vec![DVec3::ZERO; n];
        let mut specific_discharge_km2_myr = vec![0.0; n];
        for cell in 0..n {
            let start = mesh.edge_offsets[cell] as usize;
            let end = mesh.edge_offsets[cell + 1] as usize;
            // Weighted normal equations for n_e dot q = signed Q_e / w_e.
            // All geometry is promoted to f64 before accumulation.
            let (mut a_xx, mut a_xy, mut a_yy) = (0.0, 0.0, 0.0);
            let (mut b_x, mut b_y) = (0.0, 0.0);
            for edge in start..end {
                let width = f64::from(mesh.edge_face_width_km[edge]);
                let neighbor = mesh.edge_neighbor[edge] as usize;
                let normal =
                    (mesh.cell_center_km[neighbor] - mesh.cell_center_km[cell]).normalize();
                let signed_flux = directed_edge_flux_km3_myr[edge]
                    - directed_edge_flux_km3_myr[reverse_edge[edge]];
                let sample = signed_flux / width;
                a_xx += width * normal.x * normal.x;
                a_xy += width * normal.x * normal.y;
                a_yy += width * normal.y * normal.y;
                b_x += width * normal.x * sample;
                b_y += width * normal.y * sample;
            }
            if use_portals {
                for &face_index in &boundary_faces_by_cell[cell] {
                    let face = &mesh.boundary_faces[face_index];
                    if !matches!(face.condition, BoundaryFaceCondition::OpenBaseLevel { .. }) {
                        continue;
                    }
                    let width = face.width_km;
                    let normal = face.outward_normal;
                    let sample = boundary_face_flux_km3_myr[face_index] / width;
                    a_xx += width * normal.x * normal.x;
                    a_xy += width * normal.x * normal.y;
                    a_yy += width * normal.y * normal.y;
                    b_x += width * normal.x * sample;
                    b_y += width * normal.y * sample;
                }
            }
            let determinant = a_xx * a_yy - a_xy * a_xy;
            if determinant > f64::EPSILON * (a_xx + a_yy).powi(2) {
                let q = DVec3::new(
                    (a_yy * b_x - a_xy * b_y) / determinant,
                    (a_xx * b_y - a_xy * b_x) / determinant,
                    0.0,
                );
                specific_discharge_vector_km2_myr[cell] = q;
                specific_discharge_km2_myr[cell] = q.length();
            }
        }

        let total_supply_km3_myr = local_supply_km3_myr.iter().sum();
        let mut portal_outflow_km3_myr: Vec<_> = mesh
            .outlet_portals
            .iter()
            .map(|portal| (portal.id, 0.0))
            .collect();
        for (face, flux) in mesh.boundary_faces.iter().zip(&boundary_face_flux_km3_myr) {
            if let BoundaryFaceCondition::OpenBaseLevel { portal_id, .. } = face.condition {
                let (_, total) = portal_outflow_km3_myr
                    .iter_mut()
                    .find(|(id, _)| *id == portal_id)
                    .ok_or_else(|| {
                        ContinuumFlowError(format!("boundary references unknown {portal_id:?}"))
                    })?;
                *total += flux;
            }
        }
        let total_portal_outflow_km3_myr =
            portal_outflow_km3_myr.iter().map(|(_, flux)| flux).sum();
        let total_sink_storage_km3_myr = sink_storage_km3_myr.iter().sum();
        Ok(Self {
            partition,
            directed_edge_fraction,
            directed_edge_flux_km3_myr,
            boundary_face_fraction,
            boundary_face_flux_km3_myr,
            portal_outflow_km3_myr,
            routing_elevation_km,
            flat_potential,
            local_supply_km3_myr: local_supply_km3_myr.to_vec(),
            available_supply_km3_myr,
            specific_discharge_vector_km2_myr,
            specific_discharge_km2_myr,
            sink_storage_km3_myr,
            high_to_low_order,
            total_supply_km3_myr,
            total_portal_outflow_km3_myr,
            total_sink_storage_km3_myr,
        })
    }

    pub fn water_balance_error_km3_myr(&self) -> f64 {
        self.total_supply_km3_myr
            - self.total_portal_outflow_km3_myr
            - self.total_sink_storage_km3_myr
    }
}

#[derive(Debug, Clone, Copy)]
struct FillEntry {
    level: f64,
    cell: usize,
}

impl PartialEq for FillEntry {
    fn eq(&self, other: &Self) -> bool {
        self.level == other.level && self.cell == other.cell
    }
}

impl Eq for FillEntry {}

impl PartialOrd for FillEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FillEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse both keys so BinaryHeap acts as a deterministic min-heap.
        other
            .level
            .total_cmp(&self.level)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

impl DepressionRoutingSurface {
    pub fn derive(
        mesh: &LandscapeMesh,
        physical_elevation_km: &[f64],
    ) -> Result<Self, ContinuumFlowError> {
        mesh.validate()
            .map_err(|error| ContinuumFlowError(error.to_string()))?;
        let n = mesh.cell_count();
        if physical_elevation_km.len() != n {
            return Err(ContinuumFlowError(format!(
                "elevation length {}, expected {n}",
                physical_elevation_km.len()
            )));
        }
        if physical_elevation_km.iter().any(|z| !z.is_finite()) {
            return Err(ContinuumFlowError("elevation must be finite".into()));
        }

        let mut filled_elevation_km = vec![f64::INFINITY; n];
        let mut heap = BinaryHeap::new();
        let mut has_portal_seed = false;
        for face in &mesh.boundary_faces {
            if let BoundaryFaceCondition::OpenBaseLevel { elevation_km, .. } = face.condition {
                has_portal_seed = true;
                let cell = face.cell as usize;
                let level = physical_elevation_km[cell].max(f64::from(elevation_km));
                if level < filled_elevation_km[cell] {
                    filled_elevation_km[cell] = level;
                    heap.push(FillEntry { level, cell });
                }
            }
        }
        if !has_portal_seed {
            return Err(ContinuumFlowError(
                "depression routing requires at least one open portal face".into(),
            ));
        }

        // Minimax distance from genuine portal seeds: the derived fill at a
        // cell is the lowest spill elevation of any path to a portal.
        while let Some(FillEntry { level, cell }) = heap.pop() {
            if level != filled_elevation_km[cell] {
                continue;
            }
            let start = mesh.edge_offsets[cell] as usize;
            let end = mesh.edge_offsets[cell + 1] as usize;
            for edge in start..end {
                let neighbor = mesh.edge_neighbor[edge] as usize;
                let candidate = level.max(physical_elevation_km[neighbor]);
                if candidate < filled_elevation_km[neighbor] {
                    filled_elevation_km[neighbor] = candidate;
                    heap.push(FillEntry {
                        level: candidate,
                        cell: neighbor,
                    });
                }
            }
        }
        if filled_elevation_km.iter().any(|level| !level.is_finite()) {
            return Err(ContinuumFlowError(
                "one or more cells are disconnected from every portal".into(),
            ));
        }

        // Seed every equal-filled component at all of its genuine exits, then
        // compute graph distance independently of the fill heap's visit order.
        let mut flat_potential = vec![None::<u32>; n];
        let mut queue = VecDeque::new();
        let mut open_by_cell = vec![false; n];
        for face in &mesh.boundary_faces {
            if matches!(face.condition, BoundaryFaceCondition::OpenBaseLevel { .. }) {
                open_by_cell[face.cell as usize] = true;
            }
        }
        for cell in 0..n {
            let has_lower_neighbor = (mesh.edge_offsets[cell] as usize
                ..mesh.edge_offsets[cell + 1] as usize)
                .any(|edge| {
                    filled_elevation_km[mesh.edge_neighbor[edge] as usize]
                        < filled_elevation_km[cell]
                });
            if has_lower_neighbor || open_by_cell[cell] {
                flat_potential[cell] = Some(0);
                queue.push_back(cell);
            }
        }
        while let Some(cell) = queue.pop_front() {
            let next = flat_potential[cell]
                .expect("queued flat cell has a potential")
                .checked_add(1)
                .ok_or_else(|| ContinuumFlowError("flat potential exceeds u32".into()))?;
            for edge in mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize {
                let neighbor = mesh.edge_neighbor[edge] as usize;
                if flat_potential[neighbor].is_none()
                    && filled_elevation_km[neighbor] == filled_elevation_km[cell]
                {
                    flat_potential[neighbor] = Some(next);
                    queue.push_back(neighbor);
                }
            }
        }

        Ok(Self {
            filled_elevation_km,
            flat_potential,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuumFlowError(pub String);

impl fmt::Display for ContinuumFlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ContinuumFlowError {}

fn reciprocal_edges(mesh: &LandscapeMesh) -> Result<Vec<usize>, ContinuumFlowError> {
    let mut reverse = vec![usize::MAX; mesh.edge_neighbor.len()];
    for cell in 0..mesh.cell_count() {
        let start = mesh.edge_offsets[cell] as usize;
        let end = mesh.edge_offsets[cell + 1] as usize;
        for (edge, reverse_slot) in reverse.iter_mut().enumerate().take(end).skip(start) {
            let neighbor = mesh.edge_neighbor[edge] as usize;
            let reverse_edge = (mesh.edge_offsets[neighbor] as usize
                ..mesh.edge_offsets[neighbor + 1] as usize)
                .find(|&candidate| mesh.edge_neighbor[candidate] as usize == cell)
                .ok_or_else(|| ContinuumFlowError(format!("missing reciprocal edge {edge}")))?;
            *reverse_slot = reverse_edge;
        }
    }
    Ok(reverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landscape::{BoundarySide, OutletPortal};

    fn plane(mesh: &LandscapeMesh, angle: f64) -> Vec<f64> {
        let downhill = DVec3::new(angle.cos(), angle.sin(), 0.0);
        mesh.cell_center_km
            .iter()
            .map(|center| -center.dot(downhill))
            .collect()
    }

    #[test]
    fn mfd_is_conservative_and_uses_only_strictly_downhill_faces() {
        let mesh = LandscapeMesh::uniform_planar_hex(64.0, 48.0, 4.0).unwrap();
        let elevation = plane(&mesh, 0.37);
        let supply: Vec<_> = mesh.cell_area_km2.iter().map(|area| 0.2 * area).collect();
        let flow =
            FaceFlowCache::route(&mesh, &elevation, &supply, FlowPartition::MfdSlope).unwrap();

        assert!(flow.water_balance_error_km3_myr().abs() < 1e-10);
        for cell in 0..mesh.cell_count() {
            let start = mesh.edge_offsets[cell] as usize;
            let end = mesh.edge_offsets[cell + 1] as usize;
            let fraction_sum: f64 = flow.directed_edge_fraction[start..end].iter().sum();
            if fraction_sum > 0.0 {
                assert!((fraction_sum - 1.0).abs() < 1e-12);
            }
            let incoming: f64 = (start..end)
                .map(|edge| {
                    let neighbor = mesh.edge_neighbor[edge] as usize;
                    let reverse = (mesh.edge_offsets[neighbor] as usize
                        ..mesh.edge_offsets[neighbor + 1] as usize)
                        .find(|&candidate| mesh.edge_neighbor[candidate] as usize == cell)
                        .unwrap();
                    flow.directed_edge_flux_km3_myr[reverse]
                })
                .sum();
            let outgoing: f64 = flow.directed_edge_flux_km3_myr[start..end].iter().sum();
            let residual = supply[cell] + incoming - outgoing - flow.sink_storage_km3_myr[cell];
            assert!(residual.abs() < 1e-10);
            for edge in start..end {
                if flow.directed_edge_flux_km3_myr[edge] > 0.0 {
                    let neighbor = mesh.edge_neighbor[edge] as usize;
                    assert!(elevation[cell] > elevation[neighbor]);
                }
            }
        }
    }

    #[test]
    fn flats_and_local_minima_store_water_without_inventing_routes() {
        let mesh = LandscapeMesh::uniform_planar_hex(32.0, 24.0, 4.0).unwrap();
        let elevation = vec![1.0; mesh.cell_count()];
        let supply = vec![3.0; mesh.cell_count()];
        let flow =
            FaceFlowCache::route(&mesh, &elevation, &supply, FlowPartition::MfdSlope).unwrap();
        assert!(flow.directed_edge_flux_km3_myr.iter().all(|q| *q == 0.0));
        assert_eq!(flow.sink_storage_km3_myr, supply);
        assert_eq!(flow.water_balance_error_km3_myr(), 0.0);
    }

    #[test]
    fn reciprocal_faces_carry_flux_in_only_one_direction() {
        let mesh = LandscapeMesh::uniform_planar_hex(48.0, 40.0, 4.0).unwrap();
        let elevation = plane(&mesh, 1.1);
        let supply = vec![1.0; mesh.cell_count()];
        let flow =
            FaceFlowCache::route(&mesh, &elevation, &supply, FlowPartition::MfdSlope).unwrap();
        let reverse = reciprocal_edges(&mesh).unwrap();
        for (edge, &reverse_edge) in reverse.iter().enumerate() {
            assert!(
                flow.directed_edge_flux_km3_myr[edge] == 0.0
                    || flow.directed_edge_flux_km3_myr[reverse_edge] == 0.0
            );
        }
    }

    fn assert_portal_cell_balances(mesh: &LandscapeMesh, flow: &FaceFlowCache, tolerance: f64) {
        let reverse = reciprocal_edges(mesh).unwrap();
        for cell in 0..mesh.cell_count() {
            let start = mesh.edge_offsets[cell] as usize;
            let end = mesh.edge_offsets[cell + 1] as usize;
            let incoming: f64 = (start..end)
                .map(|edge| flow.directed_edge_flux_km3_myr[reverse[edge]])
                .sum();
            let internal_out: f64 = flow.directed_edge_flux_km3_myr[start..end].iter().sum();
            let boundary_out: f64 = mesh
                .boundary_faces
                .iter()
                .enumerate()
                .filter(|(_, face)| face.cell as usize == cell)
                .map(|(face_index, _)| flow.boundary_face_flux_km3_myr[face_index])
                .sum();
            let residual = flow.local_supply_km3_myr[cell] + incoming
                - internal_out
                - boundary_out
                - flow.sink_storage_km3_myr[cell];
            assert!(residual.abs() < tolerance, "cell {cell}: {residual:e}");
        }
    }

    #[test]
    fn physical_portal_faces_share_partition_and_close_water_balance() {
        let portal = OutletPortal {
            id: OutletPortalId(41),
            side: BoundarySide::South,
            span_start_km: -32.0,
            span_end_km: 32.0,
            base_level_km: 0.0,
        };
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(
            96.0,
            64.0,
            4.0,
            std::slice::from_ref(&portal),
        )
        .unwrap();
        let elevation: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| center.y + 40.0)
            .collect();
        let supply: Vec<_> = mesh.cell_area_km2.iter().map(|area| 0.1 * area).collect();
        let flow =
            FaceFlowCache::route_with_portals(&mesh, &elevation, &supply, FlowPartition::MfdSlope)
                .unwrap();

        assert_eq!(flow.portal_outflow_km3_myr.len(), 1);
        assert_eq!(flow.portal_outflow_km3_myr[0].0, portal.id);
        assert!(flow.portal_outflow_km3_myr[0].1 > 0.0);
        assert!(flow.water_balance_error_km3_myr().abs() < 1e-10);
        assert_portal_cell_balances(&mesh, &flow, 1e-10);
        for (face_index, face) in mesh.boundary_faces.iter().enumerate() {
            if matches!(face.condition, BoundaryFaceCondition::Closed) {
                assert_eq!(flow.boundary_face_fraction[face_index], 0.0);
                assert_eq!(flow.boundary_face_flux_km3_myr[face_index], 0.0);
            }
        }
    }

    fn two_portal_flow(spacing: f64) -> (Vec<OutletPortalId>, f64, f64) {
        let portals = [
            OutletPortal {
                id: OutletPortalId(7),
                side: BoundarySide::South,
                span_start_km: -44.0,
                span_end_km: -8.0,
                base_level_km: 0.0,
            },
            OutletPortal {
                id: OutletPortalId(19),
                side: BoundarySide::South,
                span_start_km: 8.0,
                span_end_km: 44.0,
                base_level_km: 0.0,
            },
        ];
        let mesh =
            LandscapeMesh::uniform_planar_hex_with_portals(96.0, 64.0, spacing, &portals).unwrap();
        let elevation: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| center.y + 40.0)
            .collect();
        // Supply follows the actual finite-volume domain represented by the
        // cells; this does not equate it to an exact rectangle.
        let supply: Vec<_> = mesh.cell_area_km2.iter().map(|area| 0.1 * area).collect();
        let flow =
            FaceFlowCache::route_with_portals(&mesh, &elevation, &supply, FlowPartition::MfdSlope)
                .unwrap();
        assert!(flow.water_balance_error_km3_myr().abs() < 1e-9);
        assert_portal_cell_balances(&mesh, &flow, 1e-9);
        let ids = flow
            .portal_outflow_km3_myr
            .iter()
            .map(|entry| entry.0)
            .collect();
        let left = flow.portal_outflow_km3_myr[0].1;
        let right = flow.portal_outflow_km3_myr[1].1;
        (ids, left, right)
    }

    #[test]
    fn two_semantic_portals_remain_stable_at_8_4_2_km() {
        let mut shares = Vec::new();
        for spacing in [8.0, 4.0, 2.0] {
            let (ids, left, right) = two_portal_flow(spacing);
            assert_eq!(ids, vec![OutletPortalId(7), OutletPortalId(19)]);
            assert!(left > 0.0 && right > 0.0);
            let share = left / (left + right);
            eprintln!(
                "two portals spacing={spacing:.0} km: left={left:.9}, right={right:.9}, left_share={share:.9}"
            );
            assert!((share - 0.5).abs() < 0.04);
            shares.push(share);
        }
        assert!((shares[2] - 0.5).abs() <= (shares[0] - 0.5).abs() + 1e-12);
    }

    #[test]
    fn exact_flat_uses_bfs_potential_without_mutating_physical_elevation() {
        let portals = [
            OutletPortal {
                id: OutletPortalId(71),
                side: BoundarySide::South,
                span_start_km: -48.0,
                span_end_km: 0.0,
                base_level_km: 1.0,
            },
            OutletPortal {
                id: OutletPortalId(72),
                side: BoundarySide::South,
                span_start_km: 0.0,
                span_end_km: 48.0,
                base_level_km: 1.0,
            },
        ];
        for spacing in [8.0, 4.0, 2.0] {
            let mesh =
                LandscapeMesh::uniform_planar_hex_with_portals(96.0, 64.0, spacing, &portals)
                    .unwrap();
            let physical = vec![1.0_f64; mesh.cell_count()];
            let before: Vec<_> = physical.iter().map(|value| value.to_ne_bytes()).collect();
            let supply: Vec<_> = mesh.cell_area_km2.iter().map(|area| 0.1 * area).collect();
            let flow = FaceFlowCache::route_with_depressions(
                &mesh,
                &physical,
                &supply,
                FlowPartition::MfdSlope,
            )
            .unwrap();
            let repeat = FaceFlowCache::route_with_depressions(
                &mesh,
                &physical,
                &supply,
                FlowPartition::MfdSlope,
            )
            .unwrap();

            assert_eq!(flow, repeat);
            assert_eq!(
                before,
                physical
                    .iter()
                    .map(|value| value.to_ne_bytes())
                    .collect::<Vec<_>>()
            );
            assert!(flow.routing_elevation_km.iter().all(|level| *level == 1.0));
            assert!(flow.flat_potential.iter().all(Option::is_some));
            assert!(flow.flat_potential.iter().any(|value| value.unwrap() > 0));
            assert_eq!(
                flow.portal_outflow_km3_myr
                    .iter()
                    .map(|entry| entry.0)
                    .collect::<Vec<_>>(),
                vec![OutletPortalId(71), OutletPortalId(72)]
            );
            assert!(flow.total_sink_storage_km3_myr < 1e-10);
            assert!(flow.water_balance_error_km3_myr().abs() < 1e-9);
            assert_portal_cell_balances(&mesh, &flow, 1e-9);

            for cell in 0..mesh.cell_count() {
                let from = flow.flat_potential[cell].unwrap();
                for edge in mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize {
                    if flow.directed_edge_flux_km3_myr[edge] > 0.0 {
                        let neighbor = mesh.edge_neighbor[edge] as usize;
                        assert!(flow.flat_potential[neighbor].unwrap() < from);
                    }
                }
            }
        }
    }

    #[test]
    fn known_outer_and_nested_bowls_fill_to_their_physical_sills() {
        let portal = OutletPortal {
            id: OutletPortalId(83),
            side: BoundarySide::South,
            span_start_km: -32.0,
            span_end_km: 32.0,
            base_level_km: 0.0,
        };
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(
            64.0,
            64.0,
            2.0,
            std::slice::from_ref(&portal),
        )
        .unwrap();
        let mut physical = vec![10.0; mesh.cell_count()];
        for (cell, center) in mesh.cell_center_km.iter().enumerate() {
            let radius = center.truncate().length();
            if radius < 18.0 {
                physical[cell] = -2.0;
            }
            if radius < 9.0 {
                physical[cell] = 4.0;
            }
            if radius < 4.0 {
                physical[cell] = -5.0;
            }
            // A physical spill corridor from the south portal into the outer bowl.
            if center.x.abs() < 2.0 && center.y <= -15.0 {
                physical[cell] = 1.0;
            }
        }
        let surface = DepressionRoutingSurface::derive(&mesh, &physical).unwrap();
        let nearest = |x: f64, y: f64| {
            mesh.cell_center_km
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = (a.x - x).powi(2) + (a.y - y).powi(2);
                    let db = (b.x - x).powi(2) + (b.y - y).powi(2);
                    da.total_cmp(&db)
                })
                .unwrap()
                .0
        };
        let inner_pit = nearest(0.0, 0.0);
        let outer_pit = nearest(0.0, -12.0);
        assert_eq!(physical[inner_pit], -5.0);
        assert_eq!(surface.filled_elevation_km[inner_pit], 4.0);
        assert_eq!(physical[outer_pit], -2.0);
        assert_eq!(surface.filled_elevation_km[outer_pit], 1.0);

        let mut supply = vec![0.0; mesh.cell_count()];
        supply[inner_pit] = 3.0;
        supply[outer_pit] = 2.0;
        let before = physical.clone();
        let flow = FaceFlowCache::route_with_depressions(
            &mesh,
            &physical,
            &supply,
            FlowPartition::MfdSlope,
        )
        .unwrap();
        assert_eq!(physical, before);
        assert!(flow.total_sink_storage_km3_myr < 1e-12);
        assert!((flow.total_portal_outflow_km3_myr - 5.0).abs() < 1e-10);
        assert!(flow.water_balance_error_km3_myr().abs() < 1e-10);
    }

    fn plane_probe_error(spacing: f64, angle: f64) -> f64 {
        let mesh = LandscapeMesh::uniform_planar_hex(192.0, 160.0, spacing).unwrap();
        let downhill = DVec3::new(angle.cos(), angle.sin(), 0.0);
        let cross = DVec3::new(-downhill.y, downhill.x, 0.0);
        let elevation = plane(&mesh, angle);
        let runoff = 0.1;
        // A fixed physical rainfall mask with a long upstream fetch and enough
        // cross-stream width that the central probe is unaffected by its sides.
        let supply: Vec<_> = mesh
            .cell_center_km
            .iter()
            .zip(&mesh.cell_area_km2)
            .map(|(center, area)| {
                let u = center.dot(downhill);
                let v = center.dot(cross);
                if (-64.0..=32.0).contains(&u) && v.abs() <= 48.0 {
                    runoff * area
                } else {
                    0.0
                }
            })
            .collect();
        let flow =
            FaceFlowCache::route(&mesh, &elevation, &supply, FlowPartition::MfdSlope).unwrap();
        let mut sum = 0.0;
        let mut count = 0;
        for (cell, center) in mesh.cell_center_km.iter().enumerate() {
            let u = center.dot(downhill);
            let v = center.dot(cross);
            if (u - 24.0).abs() <= spacing * 0.55 && v.abs() <= 12.0 {
                sum += (flow.specific_discharge_vector_km2_myr[cell] - downhill * 8.8).length();
                count += 1;
            }
        }
        assert!(count > 0);
        sum / count as f64 / 8.8
    }

    #[test]
    fn fixed_physical_plane_mask_refines_at_multiple_angles() {
        for angle in [0.0, 0.31, 0.73, 1.19] {
            let errors = [8.0, 4.0, 2.0].map(|spacing| plane_probe_error(spacing, angle));
            eprintln!("plane angle={angle:.2}: relative errors 8/4/2 km = {errors:?}");
            assert!(errors[1] < errors[0], "angle {angle}: {errors:?}");
            assert!(errors[2] < errors[1], "angle {angle}: {errors:?}");
            assert!(errors[2] < 0.02, "angle {angle}: {errors:?}");
        }
    }

    fn ridge_probe_error(spacing: f64, angle: f64) -> f64 {
        let mesh = LandscapeMesh::uniform_planar_hex(192.0, 160.0, spacing).unwrap();
        let ridge_normal = DVec3::new(angle.cos(), angle.sin(), 0.0);
        let ridge_tangent = DVec3::new(-ridge_normal.y, ridge_normal.x, 0.0);
        let elevation: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| -center.dot(ridge_normal).abs())
            .collect();
        let supply: Vec<_> = mesh
            .cell_center_km
            .iter()
            .zip(&mesh.cell_area_km2)
            .map(|(center, area)| {
                let n = center.dot(ridge_normal);
                let t = center.dot(ridge_tangent);
                if n.abs() <= 48.0 && t.abs() <= 48.0 {
                    0.1 * area
                } else {
                    0.0
                }
            })
            .collect();
        let flow =
            FaceFlowCache::route(&mesh, &elevation, &supply, FlowPartition::MfdSlope).unwrap();
        let expected_q = 2.4;
        let mut error_sum = 0.0;
        let mut count = 0;
        for (cell, center) in mesh.cell_center_km.iter().enumerate() {
            let n = center.dot(ridge_normal);
            let t = center.dot(ridge_tangent);
            if (n.abs() - 24.0).abs() <= spacing * 0.55 && t.abs() <= 12.0 {
                let expected_vector = ridge_normal * expected_q * n.signum();
                error_sum +=
                    (flow.specific_discharge_vector_km2_myr[cell] - expected_vector).length();
                count += 1;
            }
        }
        assert!(count > 0);
        error_sum / count as f64 / expected_q
    }

    #[test]
    fn fixed_physical_ridge_mask_refines_at_multiple_angles() {
        for angle in [0.0, 0.31, 0.73, 1.19] {
            let errors = [8.0, 4.0, 2.0].map(|spacing| ridge_probe_error(spacing, angle));
            eprintln!("ridge angle={angle:.2}: relative errors 8/4/2 km = {errors:?}");
            assert!(errors[1] < errors[0], "angle {angle}: {errors:?}");
            assert!(errors[2] < errors[1], "angle {angle}: {errors:?}");
            assert!(errors[2] < 0.04, "angle {angle}: {errors:?}");
        }
    }

    fn divergent_radial_errors(spacing: f64) -> (f64, f64) {
        let mesh = LandscapeMesh::uniform_planar_hex(192.0, 192.0, spacing).unwrap();
        let physical: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| -center.truncate().length())
            .collect();
        let runoff = 0.1;
        let supply_radius = 48.0;
        let probe_radius = 24.0;
        let supply: Vec<_> = mesh
            .cell_center_km
            .iter()
            .zip(&mesh.cell_area_km2)
            .map(|(center, area)| {
                if center.truncate().length() <= supply_radius {
                    runoff * area
                } else {
                    0.0
                }
            })
            .collect();
        let flow =
            FaceFlowCache::route(&mesh, &physical, &supply, FlowPartition::MfdSlope).unwrap();
        let expected_magnitude = runoff * probe_radius / 2.0;
        let mut vector_error = 0.0;
        let mut magnitude_error = 0.0;
        let mut count = 0;
        for (cell, center) in mesh.cell_center_km.iter().enumerate() {
            let radius = center.truncate().length();
            if (radius - probe_radius).abs() <= spacing * 0.55 {
                let expected = center.normalize() * expected_magnitude;
                let actual = flow.specific_discharge_vector_km2_myr[cell];
                vector_error += (actual - expected).length() / expected_magnitude;
                magnitude_error +=
                    (actual.length() - expected_magnitude).abs() / expected_magnitude;
                count += 1;
            }
        }
        assert!(count > 0);
        (vector_error / count as f64, magnitude_error / count as f64)
    }

    #[test]
    fn compact_radial_supply_has_convergent_divergent_specific_discharge() {
        let errors = [8.0, 4.0, 2.0].map(divergent_radial_errors);
        eprintln!("radial divergent relative (vector,magnitude) errors 8/4/2 km = {errors:?}");
        assert!(errors[1].0 < errors[0].0, "{errors:?}");
        assert!(errors[2].0 < errors[1].0, "{errors:?}");
        assert!(errors[1].1 < errors[0].1, "{errors:?}");
        assert!(errors[2].1 < errors[1].1, "{errors:?}");
        assert!(errors[2].0 < 0.08, "{errors:?}");
        assert!(errors[2].1 < 0.08, "{errors:?}");
    }

    fn convergent_strip_errors(spacing: f64, angle: f64) -> (f64, f64) {
        let mesh = LandscapeMesh::uniform_planar_hex(192.0, 160.0, spacing).unwrap();
        let normal = DVec3::new(angle.cos(), angle.sin(), 0.0);
        let tangent = DVec3::new(-normal.y, normal.x, 0.0);
        let physical: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| center.dot(normal).abs())
            .collect();
        let runoff = 0.1;
        let supply: Vec<_> = mesh
            .cell_center_km
            .iter()
            .zip(&mesh.cell_area_km2)
            .map(|(center, area)| {
                let n = center.dot(normal);
                let t = center.dot(tangent);
                if n.abs() <= 48.0 && t.abs() <= 48.0 {
                    runoff * area
                } else {
                    0.0
                }
            })
            .collect();
        let flow =
            FaceFlowCache::route(&mesh, &physical, &supply, FlowPartition::MfdSlope).unwrap();
        let expected_magnitude = 2.4;
        let mut vector_error = 0.0;
        let mut magnitude_error = 0.0;
        let mut count = 0;
        for (cell, center) in mesh.cell_center_km.iter().enumerate() {
            let n = center.dot(normal);
            let t = center.dot(tangent);
            if (n.abs() - 24.0).abs() <= spacing * 0.55 && t.abs() <= 12.0 {
                let expected = normal * (-n.signum() * expected_magnitude);
                let actual = flow.specific_discharge_vector_km2_myr[cell];
                vector_error += (actual - expected).length() / expected_magnitude;
                magnitude_error +=
                    (actual.length() - expected_magnitude).abs() / expected_magnitude;
                count += 1;
            }
        }
        assert!(count > 0);
        (vector_error / count as f64, magnitude_error / count as f64)
    }

    #[test]
    fn symmetric_convergent_strip_refines_away_from_its_centerline_sink() {
        for angle in [0.0, 0.31, 0.73, 1.19] {
            let errors = [8.0, 4.0, 2.0].map(|spacing| convergent_strip_errors(spacing, angle));
            eprintln!(
                "convergent strip angle={angle:.2}: relative (vector,magnitude) errors 8/4/2 km = {errors:?}"
            );
            assert!(errors[1].0 < errors[0].0, "angle {angle}: {errors:?}");
            assert!(errors[2].0 < errors[1].0, "angle {angle}: {errors:?}");
            assert!(errors[1].1 < errors[0].1, "angle {angle}: {errors:?}");
            assert!(errors[2].1 < errors[1].1, "angle {angle}: {errors:?}");
            assert!(errors[2].0 < 0.04, "angle {angle}: {errors:?}");
            assert!(errors[2].1 < 0.04, "angle {angle}: {errors:?}");
        }
    }

    #[test]
    fn centerline_ls_vector_does_not_measure_two_sided_convergent_throughput() {
        for spacing in [8.0, 4.0, 2.0] {
            let mesh = LandscapeMesh::uniform_planar_hex(192.0, 160.0, spacing).unwrap();
            let physical: Vec<_> = mesh
                .cell_center_km
                .iter()
                .map(|center| center.x.abs())
                .collect();
            let supply: Vec<_> = mesh
                .cell_center_km
                .iter()
                .zip(&mesh.cell_area_km2)
                .map(|(center, area)| {
                    if center.x.abs() <= 48.0 && center.y.abs() <= 48.0 {
                        0.1 * area
                    } else {
                        0.0
                    }
                })
                .collect();
            let flow =
                FaceFlowCache::route(&mesh, &physical, &supply, FlowPartition::MfdSlope).unwrap();
            let mut sink_throughput = 0.0;
            let mut vector_magnitude = 0.0;
            let mut count = 0;
            for (cell, center) in mesh.cell_center_km.iter().enumerate() {
                if center.x.abs() <= spacing * 0.3
                    && center.y.abs() <= 12.0
                    && flow.sink_storage_km3_myr[cell] > 0.0
                {
                    // Consecutive zig-zag centerline cells are one hex-row
                    // step apart along y.
                    let contour_length = spacing * 3.0_f64.sqrt() * 0.5;
                    sink_throughput += flow.sink_storage_km3_myr[cell] / contour_length;
                    vector_magnitude += flow.specific_discharge_km2_myr[cell];
                    count += 1;
                }
            }
            assert!(count > 0);
            let mean_sink_throughput = sink_throughput / count as f64;
            let mean_vector_magnitude = vector_magnitude / count as f64;
            eprintln!(
                "centerline spacing={spacing:.0} km: sink throughput/length={mean_sink_throughput:.9}, LS |q|={mean_vector_magnitude:.9}, ratio={:.9}",
                mean_vector_magnitude / mean_sink_throughput
            );
            assert!(mean_sink_throughput > 0.0);
            assert!(mean_vector_magnitude.is_finite());
            // The vector reconstruction cancels opposing inflows and therefore
            // cannot serve as the scalar throughput diagnostic on the sink line.
            assert!(mean_vector_magnitude / mean_sink_throughput < 0.25);
            if spacing == 2.0 {
                assert!((mean_sink_throughput - 9.6).abs() < 1e-6);
            }
        }
    }

    fn broad_reach_contract_errors(spacing: f64, angle: f64) -> (f64, f64, f64) {
        let mesh = LandscapeMesh::uniform_planar_hex(256.0, 224.0, spacing).unwrap();
        let downstream = DVec3::new(angle.cos(), angle.sin(), 0.0);
        let cross = DVec3::new(-downstream.y, downstream.x, 0.0);
        let reach_half_width = 12.0;
        let reach_width = 2.0 * reach_half_width;
        let axial_slope = 0.05;
        let physical: Vec<_> = mesh
            .cell_center_km
            .iter()
            .map(|center| {
                let t = center.dot(downstream);
                let n = center.dot(cross);
                (n.abs() - reach_half_width).max(0.0) - axial_slope * t
            })
            .collect();
        let supply: Vec<_> = mesh
            .cell_center_km
            .iter()
            .zip(&mesh.cell_area_km2)
            .map(|(center, area)| {
                let t = center.dot(downstream);
                let n = center.dot(cross);
                if (-64.0..=-32.0).contains(&t) && n.abs() <= 48.0 {
                    0.1 * area
                } else {
                    0.0
                }
            })
            .collect();
        let total_supply: f64 = supply.iter().sum();
        let flow =
            FaceFlowCache::route(&mesh, &physical, &supply, FlowPartition::MfdSlope).unwrap();
        let cut_t = 32.0;
        let mut cut_flux = 0.0;
        for cell in 0..mesh.cell_count() {
            let cell_t = mesh.cell_center_km[cell].dot(downstream);
            for edge in mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize {
                let neighbor = mesh.edge_neighbor[edge] as usize;
                let neighbor_t = mesh.cell_center_km[neighbor].dot(downstream);
                if cell_t < cut_t && neighbor_t >= cut_t {
                    cut_flux += flow.directed_edge_flux_km3_myr[edge];
                }
            }
        }

        let mut vector = DVec3::ZERO;
        let mut magnitude = 0.0;
        let mut sample_area = 0.0;
        for (cell, center) in mesh.cell_center_km.iter().enumerate() {
            let t = center.dot(downstream);
            let n = center.dot(cross);
            // Area-average over a fixed physical reach window rather than a
            // resolution-dependent one-cell sampling line.
            if (24.0..=40.0).contains(&t) && n.abs() <= reach_half_width {
                let area = mesh.cell_area_km2[cell];
                vector += flow.specific_discharge_vector_km2_myr[cell] * area;
                magnitude += flow.specific_discharge_km2_myr[cell] * area;
                sample_area += area;
            }
        }
        assert!(sample_area > 0.0);
        let mean_vector = vector / sample_area;
        let mean_magnitude = magnitude / sample_area;
        // The center-inclusion mask is a finite-volume approximation to the
        // fixed 24 km reach. Use its represented support at each resolution;
        // this width converges to `reach_width` instead of assuming cells tile
        // the exact analytic rectangle at coarse resolution.
        let represented_width = sample_area / (40.0 - 24.0);
        debug_assert!((represented_width - reach_width).abs() < 2.0 * spacing);
        let expected_q = total_supply / represented_width;
        (
            (cut_flux - total_supply).abs() / total_supply,
            (mean_vector - downstream * expected_q).length() / expected_q,
            (mean_magnitude - expected_q).abs() / expected_q,
        )
    }

    #[test]
    fn broad_axial_reach_passes_cut_flux_and_local_q_contract() {
        for angle in [0.0, 0.73] {
            let errors = [8.0, 4.0, 2.0].map(|spacing| broad_reach_contract_errors(spacing, angle));
            eprintln!(
                "broad reach angle={angle:.2}: relative (cut,vector,magnitude) errors 8/4/2 km = {errors:?}"
            );
            assert!(errors[2].0 < 1e-10, "angle {angle}: {errors:?}");
            // Require net refinement of the combined local diagnostic and a
            // tight finest-grid envelope for both fields. The 4 km support can
            // align unusually exactly, so sub-percent quadrature noise need
            // not be monotone in vector and magnitude independently.
            let combined = errors.map(|error| error.1.max(error.2));
            assert!(combined[2] < combined[0], "angle {angle}: {errors:?}");
            assert!(errors[2].1 < 0.005, "angle {angle}: {errors:?}");
            assert!(errors[2].2 < 0.005, "angle {angle}: {errors:?}");
        }
    }
}
