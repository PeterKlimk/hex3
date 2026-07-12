use glam::{DVec3, Vec3};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Per-cell treatment when routing or material flux reaches the domain edge.
///
/// North/south edge cells are open outlets in the standard patch. East/west
/// edge cells are closed. At a corner, the open outlet takes precedence.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BoundaryCondition {
    Interior,
    Closed,
    OpenBaseLevel { elevation_km: f32 },
}

/// Stable identity for one physical outlet segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutletPortalId(pub u32);

/// Side of the nominal rectangular test domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundarySide {
    North,
    South,
    East,
    West,
}

/// A physical outlet segment whose geometry does not depend on mesh spacing.
///
/// `span_*_km` is x on north/south sides and y on east/west sides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutletPortal {
    pub id: OutletPortalId,
    pub side: BoundarySide,
    pub span_start_km: f64,
    pub span_end_km: f64,
    pub base_level_km: f32,
}

impl OutletPortal {
    pub fn width_km(&self) -> f64 {
        self.span_end_km - self.span_start_km
    }
}

/// Boundary condition on one physical boundary-face segment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BoundaryFaceCondition {
    Closed,
    OpenBaseLevel {
        portal_id: OutletPortalId,
        elevation_km: f32,
    },
}

/// A segment of an exposed face of one full hexagonal cell.
///
/// `width_km` is physical length along the (possibly sawtooth) boundary and is
/// the width used by finite-volume fluxes. `projected_span_*_km` is the
/// segment's projection onto the coordinate used by its nominal side (x for
/// north/south, y for east/west). Portal endpoints split physical faces, so
/// projected portal coverage and physical boundary arc length remain explicit
/// and must not be treated as interchangeable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandscapeBoundaryFace {
    pub cell: u32,
    pub side: BoundarySide,
    pub center_km: DVec3,
    pub outward_normal: DVec3,
    pub width_km: f64,
    pub projected_span_start_km: f64,
    pub projected_span_end_km: f64,
    pub center_distance_km: f64,
    pub condition: BoundaryFaceCondition,
}

impl LandscapeBoundaryFace {
    /// Width of this face segment after projection onto its nominal side.
    pub fn projected_width_km(&self) -> f64 {
        self.projected_span_end_km - self.projected_span_start_km
    }
}

/// Geometry-only finite-volume graph shared by all landscape operators.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LandscapeMesh {
    pub cell_center_km: Vec<DVec3>,
    pub cell_area_km2: Vec<f64>,
    pub edge_offsets: Vec<u32>,
    pub edge_neighbor: Vec<u32>,
    pub edge_distance_km: Vec<f32>,
    pub edge_face_width_km: Vec<f32>,
    pub edge_outward_tangent: Vec<Vec3>,
    /// Compatibility cell view used by the Slice 1 solver. New boundary
    /// operators should consume `boundary_faces` instead.
    pub boundary: Vec<BoundaryCondition>,
    pub boundary_faces: Vec<LandscapeBoundaryFace>,
    pub outlet_portals: Vec<OutletPortal>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LandscapeMeshError(pub String);

impl fmt::Display for LandscapeMeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LandscapeMeshError {}

impl LandscapeMesh {
    /// Construct a centered, pointy-row planar hex patch.
    ///
    /// `spacing_km` is the center-to-center distance between neighbors. The
    /// requested width and height are physical extents; the nearest whole
    /// number of rows and columns is used symmetrically around the origin.
    pub fn uniform_planar_hex(
        width_km: f64,
        height_km: f64,
        spacing_km: f64,
    ) -> Result<Self, LandscapeMeshError> {
        let portals = [
            OutletPortal {
                id: OutletPortalId(0),
                side: BoundarySide::South,
                span_start_km: -0.5 * width_km,
                span_end_km: 0.5 * width_km,
                base_level_km: 0.0,
            },
            OutletPortal {
                id: OutletPortalId(1),
                side: BoundarySide::North,
                span_start_km: -0.5 * width_km,
                span_end_km: 0.5 * width_km,
                base_level_km: 0.0,
            },
        ];
        Self::uniform_planar_hex_with_portals(width_km, height_km, spacing_km, &portals)
    }

    /// Construct a planar patch with outlet portals fixed in physical space.
    ///
    /// Boundary portions outside the supplied portals are closed. Portal
    /// identity and extent are copied verbatim; refinement only changes which
    /// cells own the resulting boundary-face segments.
    pub fn uniform_planar_hex_with_portals(
        width_km: f64,
        height_km: f64,
        spacing_km: f64,
        outlet_portals: &[OutletPortal],
    ) -> Result<Self, LandscapeMeshError> {
        if !width_km.is_finite() || !height_km.is_finite() || !spacing_km.is_finite() {
            return Err(LandscapeMeshError("mesh dimensions must be finite".into()));
        }
        if width_km <= 0.0 || height_km <= 0.0 || spacing_km <= 0.0 {
            return Err(LandscapeMeshError(
                "mesh dimensions must be positive".into(),
            ));
        }
        validate_portals(width_km, height_km, outlet_portals)?;

        let row_step = spacing_km * 3.0_f64.sqrt() * 0.5;
        let cols = (width_km / spacing_km).round().max(2.0) as usize;
        let rows = (height_km / row_step).round().max(2.0) as usize;
        let n = rows
            .checked_mul(cols)
            .ok_or_else(|| LandscapeMeshError("mesh cell count overflows address space".into()))?;
        if n > u32::MAX as usize {
            return Err(LandscapeMeshError("mesh exceeds u32 graph indexing".into()));
        }

        let mut cell_center_km = Vec::with_capacity(n);
        let y0 = -0.5 * (rows.saturating_sub(1) as f64) * row_step;
        for row in 0..rows {
            // Stagger adjacent rows by half a spacing while keeping the patch
            // centered as a whole (even rows at -s/4, odd rows at +s/4).
            let row_offset = if row % 2 == 0 {
                -0.25 * spacing_km
            } else {
                0.25 * spacing_km
            };
            let x0 = -0.5 * (cols.saturating_sub(1) as f64) * spacing_km + row_offset;
            for col in 0..cols {
                cell_center_km.push(DVec3::new(
                    x0 + col as f64 * spacing_km,
                    y0 + row as f64 * row_step,
                    0.0,
                ));
            }
        }

        let cell_area = 3.0_f64.sqrt() * 0.5 * spacing_km * spacing_km;
        let face_width = spacing_km / 3.0_f64.sqrt();
        let cell_area_km2 = vec![cell_area; n];
        let mut edge_offsets = Vec::with_capacity(n + 1);
        let mut edge_neighbor = Vec::with_capacity(n * 6);
        let mut edge_distance_km = Vec::with_capacity(n * 6);
        let mut edge_face_width_km = Vec::with_capacity(n * 6);
        let mut edge_outward_tangent = Vec::with_capacity(n * 6);
        let mut boundary = Vec::with_capacity(n);

        for row in 0..rows {
            for col in 0..cols {
                let i = row * cols + col;
                edge_offsets.push(edge_neighbor.len() as u32);
                // Odd-row offset coordinates. Keeping a fixed order makes the
                // graph and all downstream reductions deterministic.
                let neighbors: [(isize, isize); 6] = if row % 2 == 0 {
                    [(-1, 0), (1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]
                } else {
                    [(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]
                };
                for (dc, dr) in neighbors {
                    let nc = col as isize + dc;
                    let nr = row as isize + dr;
                    if nc < 0 || nr < 0 || nc >= cols as isize || nr >= rows as isize {
                        continue;
                    }
                    let j = nr as usize * cols + nc as usize;
                    let delta = cell_center_km[j] - cell_center_km[i];
                    let distance = delta.length();
                    edge_neighbor.push(j as u32);
                    edge_distance_km.push(distance as f32);
                    edge_face_width_km.push(face_width as f32);
                    edge_outward_tangent.push(Vec3::new(
                        (delta.x / distance) as f32,
                        (delta.y / distance) as f32,
                        0.0,
                    ));
                }
                boundary.push(
                    if row == 0 || row + 1 == rows || col == 0 || col + 1 == cols {
                        BoundaryCondition::Closed
                    } else {
                        BoundaryCondition::Interior
                    },
                );
            }
        }
        edge_offsets.push(edge_neighbor.len() as u32);

        let boundary_faces =
            build_boundary_faces(spacing_km, rows, cols, &cell_center_km, outlet_portals);
        // Preserve the Slice 1 cell-level view without inventing a distinct
        // outlet per cell. A cell is open when any of its physical subfaces is
        // assigned to a portal; new operators retain the portal identity by
        // consuming `boundary_faces` directly.
        for face in &boundary_faces {
            if let BoundaryFaceCondition::OpenBaseLevel { elevation_km, .. } = face.condition {
                boundary[face.cell as usize] = BoundaryCondition::OpenBaseLevel { elevation_km };
            }
        }

        let mesh = Self {
            cell_center_km,
            cell_area_km2,
            edge_offsets,
            edge_neighbor,
            edge_distance_km,
            edge_face_width_km,
            edge_outward_tangent,
            boundary,
            boundary_faces,
            outlet_portals: outlet_portals.to_vec(),
        };
        mesh.validate()?;
        Ok(mesh)
    }

    pub fn cell_count(&self) -> usize {
        self.cell_center_km.len()
    }

    /// Area of the actual finite-volume domain: the union of all full cells.
    pub fn actual_domain_area_km2(&self) -> f64 {
        self.cell_area_km2.iter().sum()
    }

    /// Physical length of the exposed, generally sawtooth, cell boundary.
    pub fn actual_boundary_arc_length_km(&self) -> f64 {
        self.boundary_faces.iter().map(|face| face.width_km).sum()
    }

    /// Coordinate-projected coverage assigned to one semantic portal.
    ///
    /// This is comparable to [`OutletPortal::width_km`]; it is deliberately
    /// distinct from the physical arc length used by finite-volume fluxes.
    pub fn portal_projected_coverage_km(&self, portal_id: OutletPortalId) -> f64 {
        self.boundary_faces
            .iter()
            .filter(|face| {
                matches!(
                    face.condition,
                    BoundaryFaceCondition::OpenBaseLevel { portal_id: id, .. }
                        if id == portal_id
                )
            })
            .map(LandscapeBoundaryFace::projected_width_km)
            .sum()
    }

    pub fn validate(&self) -> Result<(), LandscapeMeshError> {
        let n = self.cell_count();
        if n == 0 {
            return Err(LandscapeMeshError("mesh is empty".into()));
        }
        if self.cell_area_km2.len() != n || self.boundary.len() != n {
            return Err(LandscapeMeshError("cell-array length mismatch".into()));
        }
        for portal in &self.outlet_portals {
            if !portal.span_start_km.is_finite()
                || !portal.span_end_km.is_finite()
                || portal.span_start_km >= portal.span_end_km
                || !portal.base_level_km.is_finite()
            {
                return Err(LandscapeMeshError(format!(
                    "invalid outlet portal {:?}",
                    portal.id
                )));
            }
        }
        for face in &self.boundary_faces {
            if face.cell as usize >= n
                || !face.center_km.is_finite()
                || !face.outward_normal.is_finite()
                || !face.width_km.is_finite()
                || face.width_km <= 0.0
                || !face.projected_span_start_km.is_finite()
                || !face.projected_span_end_km.is_finite()
                || face.projected_span_start_km >= face.projected_span_end_km
                || !face.center_distance_km.is_finite()
                || face.center_distance_km <= 0.0
            {
                return Err(LandscapeMeshError("invalid boundary-face geometry".into()));
            }
            if let BoundaryFaceCondition::OpenBaseLevel {
                portal_id,
                elevation_km,
            } = face.condition
            {
                if !elevation_km.is_finite()
                    || !self
                        .outlet_portals
                        .iter()
                        .any(|portal| portal.id == portal_id && portal.side == face.side)
                {
                    return Err(LandscapeMeshError(format!(
                        "invalid portal assignment on boundary cell {}",
                        face.cell
                    )));
                }
            }
        }
        if self.edge_offsets.len() != n + 1 || self.edge_offsets[0] != 0 {
            return Err(LandscapeMeshError("invalid CSR offsets".into()));
        }
        let m = self.edge_neighbor.len();
        if self.edge_distance_km.len() != m
            || self.edge_face_width_km.len() != m
            || self.edge_outward_tangent.len() != m
            || self.edge_offsets[n] as usize != m
        {
            return Err(LandscapeMeshError("edge-array length mismatch".into()));
        }
        for pair in self.edge_offsets.windows(2) {
            if pair[0] > pair[1] {
                return Err(LandscapeMeshError("CSR offsets are not monotone".into()));
            }
        }
        for (i, center) in self.cell_center_km.iter().enumerate() {
            if !center.is_finite()
                || !self.cell_area_km2[i].is_finite()
                || self.cell_area_km2[i] <= 0.0
            {
                return Err(LandscapeMeshError(format!("invalid cell geometry at {i}")));
            }
            if let BoundaryCondition::OpenBaseLevel { elevation_km } = self.boundary[i] {
                if !elevation_km.is_finite() {
                    return Err(LandscapeMeshError(format!("non-finite base level at {i}")));
                }
            }
            let start = self.edge_offsets[i] as usize;
            let end = self.edge_offsets[i + 1] as usize;
            for edge in start..end {
                let j = self.edge_neighbor[edge] as usize;
                if j >= n || j == i {
                    return Err(LandscapeMeshError(format!(
                        "invalid neighbor at edge {edge}"
                    )));
                }
                if !self.edge_distance_km[edge].is_finite()
                    || self.edge_distance_km[edge] <= 0.0
                    || !self.edge_face_width_km[edge].is_finite()
                    || self.edge_face_width_km[edge] <= 0.0
                    || !self.edge_outward_tangent[edge].is_finite()
                {
                    return Err(LandscapeMeshError(format!(
                        "invalid face geometry at edge {edge}"
                    )));
                }
                let reverse_start = self.edge_offsets[j] as usize;
                let reverse_end = self.edge_offsets[j + 1] as usize;
                if !(reverse_start..reverse_end).any(|r| self.edge_neighbor[r] as usize == i) {
                    return Err(LandscapeMeshError(format!("asymmetric adjacency {i}->{j}")));
                }
            }
        }
        Ok(())
    }
}

fn validate_portals(
    width_km: f64,
    height_km: f64,
    portals: &[OutletPortal],
) -> Result<(), LandscapeMeshError> {
    for (i, portal) in portals.iter().enumerate() {
        let half_span = match portal.side {
            BoundarySide::North | BoundarySide::South => 0.5 * width_km,
            BoundarySide::East | BoundarySide::West => 0.5 * height_km,
        };
        if !portal.span_start_km.is_finite()
            || !portal.span_end_km.is_finite()
            || portal.span_start_km >= portal.span_end_km
            || portal.span_start_km < -half_span
            || portal.span_end_km > half_span
            || !portal.base_level_km.is_finite()
        {
            return Err(LandscapeMeshError(format!("invalid outlet portal {i}")));
        }
        if portals[..i].iter().any(|other| other.id == portal.id) {
            return Err(LandscapeMeshError(format!(
                "duplicate outlet portal id {:?}",
                portal.id
            )));
        }
        if portals[..i].iter().any(|other| {
            other.side == portal.side
                && other.span_start_km < portal.span_end_km
                && portal.span_start_km < other.span_end_km
        }) {
            return Err(LandscapeMeshError(
                "outlet portal segments may not overlap".into(),
            ));
        }
    }
    Ok(())
}

fn build_boundary_faces(
    spacing_km: f64,
    rows: usize,
    cols: usize,
    centers: &[DVec3],
    portals: &[OutletPortal],
) -> Vec<LandscapeBoundaryFace> {
    let face_width = spacing_km / 3.0_f64.sqrt();
    let mut faces = Vec::with_capacity(4 * (rows + cols) + 2 * portals.len());

    for row in 0..rows {
        let neighbors: [(isize, isize); 6] = if row % 2 == 0 {
            [(-1, 0), (1, 0), (-1, -1), (0, -1), (-1, 1), (0, 1)]
        } else {
            [(-1, 0), (1, 0), (0, -1), (1, -1), (0, 1), (1, 1)]
        };
        for col in 0..cols {
            let cell = row * cols + col;
            for (dc, dr) in neighbors {
                let nc = col as isize + dc;
                let nr = row as isize + dr;
                if nc >= 0 && nr >= 0 && nc < cols as isize && nr < rows as isize {
                    continue;
                }

                // A missing lattice neighbor identifies one and only one
                // exposed regular-hex face. At a corner, row escape owns the
                // diagonal face and column escape owns the horizontal face.
                let side = if nr < 0 {
                    BoundarySide::South
                } else if nr >= rows as isize {
                    BoundarySide::North
                } else if nc < 0 {
                    BoundarySide::West
                } else {
                    BoundarySide::East
                };
                let virtual_center =
                    virtual_neighbor_center(centers[cell], row, dc, dr, spacing_km);
                let delta = virtual_center - centers[cell];
                let normal_d = delta / spacing_km;
                let tangent = DVec3::new(-normal_d.y, normal_d.x, 0.0);
                let raw_center = centers[cell] + 0.5 * delta;
                split_boundary_face(
                    &mut faces,
                    cell,
                    side,
                    raw_center,
                    normal_d,
                    tangent,
                    face_width,
                    0.5 * spacing_km,
                    portals,
                );
            }
        }
    }
    faces
}

fn virtual_neighbor_center(
    center: DVec3,
    row: usize,
    dc: isize,
    dr: isize,
    spacing_km: f64,
) -> DVec3 {
    if dr == 0 {
        return center + DVec3::new(dc as f64 * spacing_km, 0.0, 0.0);
    }
    let row_step = spacing_km * 3.0_f64.sqrt() * 0.5;
    let horizontal = if row.is_multiple_of(2) {
        (dc as f64 + 0.5) * spacing_km
    } else {
        (dc as f64 - 0.5) * spacing_km
    };
    center + DVec3::new(horizontal, dr as f64 * row_step, 0.0)
}

#[allow(clippy::too_many_arguments)]
fn split_boundary_face(
    output: &mut Vec<LandscapeBoundaryFace>,
    cell: usize,
    side: BoundarySide,
    center: DVec3,
    normal: DVec3,
    tangent: DVec3,
    width_km: f64,
    center_distance_km: f64,
    portals: &[OutletPortal],
) {
    let endpoint_a = center - 0.5 * width_km * tangent;
    let endpoint_b = center + 0.5 * width_km * tangent;
    let projected = |point: DVec3| match side {
        BoundarySide::North | BoundarySide::South => point.x,
        BoundarySide::East | BoundarySide::West => point.y,
    };
    let pa = projected(endpoint_a);
    let pb = projected(endpoint_b);
    debug_assert!((pb - pa).abs() > f64::EPSILON);

    let mut cuts = vec![0.0, 1.0];
    let pmin = pa.min(pb);
    let pmax = pa.max(pb);
    for portal in portals.iter().filter(|portal| portal.side == side) {
        for cut in [portal.span_start_km, portal.span_end_km] {
            if cut > pmin && cut < pmax {
                cuts.push((cut - pa) / (pb - pa));
            }
        }
    }
    cuts.sort_by(f64::total_cmp);
    cuts.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);

    for cut_pair in cuts.windows(2) {
        let u0 = cut_pair[0];
        let u1 = cut_pair[1];
        let segment_center = endpoint_a + (0.5 * (u0 + u1)) * (endpoint_b - endpoint_a);
        let projected_a = pa + u0 * (pb - pa);
        let projected_b = pa + u1 * (pb - pa);
        let projected_midpoint = 0.5 * (projected_a + projected_b);
        let condition = portals
            .iter()
            .find(|portal| {
                portal.side == side
                    && projected_midpoint >= portal.span_start_km
                    && projected_midpoint <= portal.span_end_km
            })
            .map_or(BoundaryFaceCondition::Closed, |portal| {
                BoundaryFaceCondition::OpenBaseLevel {
                    portal_id: portal.id,
                    elevation_km: portal.base_level_km,
                }
            });
        output.push(LandscapeBoundaryFace {
            cell: cell as u32,
            side,
            center_km: segment_center,
            outward_normal: normal,
            width_km: (u1 - u0) * width_km,
            projected_span_start_km: projected_a.min(projected_b),
            projected_span_end_km: projected_a.max(projected_b),
            center_distance_km,
            condition,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_patch_is_valid_and_has_declared_boundaries() {
        let mesh = LandscapeMesh::uniform_planar_hex(960.0, 640.0, 8.0).unwrap();
        mesh.validate().unwrap();
        assert!(mesh.cell_count() > 10_000);
        assert!(mesh
            .boundary
            .iter()
            .any(|b| matches!(b, BoundaryCondition::Closed)));
        assert!(mesh
            .boundary
            .iter()
            .any(|b| matches!(b, BoundaryCondition::OpenBaseLevel { .. })));
        assert!(mesh
            .boundary
            .iter()
            .any(|b| matches!(b, BoundaryCondition::Interior)));
        for portal in &mesh.outlet_portals {
            let projected_coverage = mesh.portal_projected_coverage_km(portal.id);
            assert!(
                projected_coverage <= portal.width_km()
                    && portal.width_km() - projected_coverage < 8.0
            );
        }
    }

    #[test]
    fn every_internal_face_is_reciprocal_and_uniform() {
        let mesh = LandscapeMesh::uniform_planar_hex(48.0, 40.0, 4.0).unwrap();
        for i in 0..mesh.cell_count() {
            for edge in mesh.edge_offsets[i] as usize..mesh.edge_offsets[i + 1] as usize {
                assert!((mesh.edge_distance_km[edge] - 4.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn rejects_invalid_dimensions() {
        assert!(LandscapeMesh::uniform_planar_hex(10.0, 10.0, 0.0).is_err());
        assert!(LandscapeMesh::uniform_planar_hex(f64::NAN, 10.0, 1.0).is_err());
    }

    #[test]
    fn physical_portals_retain_identity_and_span_under_refinement() {
        let portals = [
            OutletPortal {
                id: OutletPortalId(17),
                side: BoundarySide::South,
                span_start_km: -32.0,
                span_end_km: 32.0,
                base_level_km: 0.125,
            },
            OutletPortal {
                id: OutletPortalId(23),
                side: BoundarySide::North,
                span_start_km: 120.0,
                span_end_km: 168.0,
                base_level_km: 0.25,
            },
        ];

        let mut open_face_counts = Vec::new();
        for spacing in [8.0, 4.0, 2.0] {
            let mesh =
                LandscapeMesh::uniform_planar_hex_with_portals(960.0, 640.0, spacing, &portals)
                    .unwrap();
            assert_eq!(mesh.outlet_portals, portals);
            assert_eq!(mesh.outlet_portals.len(), 2);

            let mut count = 0;
            for portal in &portals {
                let faces: Vec<_> = mesh
                    .boundary_faces
                    .iter()
                    .filter(|face| {
                        matches!(
                            face.condition,
                            BoundaryFaceCondition::OpenBaseLevel { portal_id, .. }
                                if portal_id == portal.id
                        )
                    })
                    .collect();
                count += faces.len();
                let projected_coverage: f64 =
                    faces.iter().map(|face| face.projected_width_km()).sum();
                assert!((projected_coverage - portal.width_km()).abs() < 1e-10);
                assert!(faces.iter().all(|face| face.side == portal.side));
            }
            open_face_counts.push(count);
        }

        // Refinement changes ownership granularity, not semantic outlets.
        assert!(open_face_counts[0] < open_face_counts[1]);
        assert!(open_face_counts[1] < open_face_counts[2]);
    }

    #[test]
    fn exposed_faces_close_each_hexagonal_control_volume() {
        let portal = OutletPortal {
            id: OutletPortalId(5),
            side: BoundarySide::North,
            span_start_km: -17.0,
            span_end_km: 29.0,
            base_level_km: 0.0,
        };
        let mesh =
            LandscapeMesh::uniform_planar_hex_with_portals(96.0, 64.0, 4.0, &[portal]).unwrap();
        let expected_area = 0.5 * 3.0_f64.sqrt() * 4.0 * 4.0;

        assert!(mesh.boundary_faces.iter().all(|face| {
            let cell = face.cell as usize;
            (face.center_distance_km
                - (face.center_km - mesh.cell_center_km[cell])
                    .dot(face.outward_normal)
                    .abs())
            .abs()
                < 1e-6
        }));

        for cell in 0..mesh.cell_count() {
            let mut gauss = DVec3::ZERO;
            let mut area_moment = 0.0;
            for edge in mesh.edge_offsets[cell] as usize..mesh.edge_offsets[cell + 1] as usize {
                let normal = mesh.edge_outward_tangent[edge].as_dvec3();
                let width = mesh.edge_face_width_km[edge] as f64;
                gauss += width * normal;
                area_moment += width * 0.5 * mesh.edge_distance_km[edge] as f64;
            }
            for face in mesh
                .boundary_faces
                .iter()
                .filter(|face| face.cell as usize == cell)
            {
                gauss += face.width_km * face.outward_normal;
                area_moment += face.width_km
                    * (face.center_km - mesh.cell_center_km[cell]).dot(face.outward_normal);
            }
            assert!(
                gauss.length() < 2e-6,
                "cell {cell} Gauss residual {gauss:?}"
            );
            assert!(
                (0.5 * area_moment - expected_area).abs() < 2e-6,
                "cell {cell} area moment {}",
                0.5 * area_moment
            );
        }

        // The actual union of full cells, rather than the requested nominal
        // rectangle, is the finite-volume domain.
        let boundary_area_moment: f64 = mesh
            .boundary_faces
            .iter()
            .map(|face| face.width_km * face.center_km.dot(face.outward_normal))
            .sum();
        let actual_area: f64 = mesh.cell_area_km2.iter().sum();
        assert!(
            (0.5 * boundary_area_moment - actual_area).abs() < 1e-8,
            "boundary area {} != cell area {actual_area}",
            0.5 * boundary_area_moment
        );
        assert!((actual_area - mesh.cell_count() as f64 * expected_area).abs() < 1e-8);

        let actual_arc_length: f64 = mesh.boundary_faces.iter().map(|face| face.width_km).sum();
        let projected_side_coverage: f64 = mesh
            .boundary_faces
            .iter()
            .map(LandscapeBoundaryFace::projected_width_km)
            .sum();
        assert!(actual_arc_length > projected_side_coverage);
    }

    #[test]
    fn every_missing_neighbor_produces_exactly_one_unsplit_face() {
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(96.0, 64.0, 4.0, &[]).unwrap();
        let directed_internal_faces = mesh.edge_neighbor.len();
        let expected_missing_faces = 6 * mesh.cell_count() - directed_internal_faces;
        assert_eq!(mesh.boundary_faces.len(), expected_missing_faces);

        let expected_width = 4.0 / 3.0_f64.sqrt();
        for face in &mesh.boundary_faces {
            assert!((face.width_km - expected_width).abs() < 1e-12);
            assert!((face.center_distance_km - 2.0).abs() < 1e-12);
            assert!((face.outward_normal.length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn portal_splits_are_nonoverlapping_and_preserve_physical_faces() {
        let portals = [
            OutletPortal {
                id: OutletPortalId(3),
                side: BoundarySide::West,
                span_start_km: -17.3,
                span_end_km: -2.1,
                base_level_km: 0.0,
            },
            OutletPortal {
                id: OutletPortalId(4),
                side: BoundarySide::West,
                span_start_km: 1.7,
                span_end_km: 19.2,
                base_level_km: 0.0,
            },
        ];
        let unsplit = LandscapeMesh::uniform_planar_hex_with_portals(96.0, 64.0, 4.0, &[]).unwrap();
        let split =
            LandscapeMesh::uniform_planar_hex_with_portals(96.0, 64.0, 4.0, &portals).unwrap();

        let unsplit_arc: f64 = unsplit
            .boundary_faces
            .iter()
            .map(|face| face.width_km)
            .sum();
        let split_arc: f64 = split.boundary_faces.iter().map(|face| face.width_km).sum();
        assert!((unsplit_arc - split_arc).abs() < 1e-10);
        for portal in &portals {
            let coverage: f64 = split
                .boundary_faces
                .iter()
                .filter(|face| {
                    matches!(
                        face.condition,
                        BoundaryFaceCondition::OpenBaseLevel { portal_id, .. }
                            if portal_id == portal.id
                    )
                })
                .map(LandscapeBoundaryFace::projected_width_km)
                .sum();
            assert!((coverage - portal.width_km()).abs() < 1e-10);
        }
    }

    #[test]
    fn rejects_duplicate_or_overlapping_portals() {
        let portal = OutletPortal {
            id: OutletPortalId(1),
            side: BoundarySide::North,
            span_start_km: -10.0,
            span_end_km: 10.0,
            base_level_km: 0.0,
        };
        let duplicate = OutletPortal {
            side: BoundarySide::South,
            ..portal.clone()
        };
        assert!(LandscapeMesh::uniform_planar_hex_with_portals(
            100.0,
            80.0,
            4.0,
            &[portal.clone(), duplicate],
        )
        .is_err());
        let overlap = OutletPortal {
            id: OutletPortalId(2),
            span_start_km: 5.0,
            span_end_km: 15.0,
            ..portal.clone()
        };
        assert!(LandscapeMesh::uniform_planar_hex_with_portals(
            100.0,
            80.0,
            4.0,
            &[portal, overlap],
        )
        .is_err());
    }
}
