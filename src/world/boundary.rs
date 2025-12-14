//! Plate boundary extraction and classification.
//!
//! This module builds an explicit list of plate-boundary edges (between adjacent
//! tessellation cells with different plate assignments), and classifies each edge
//! by kinematic convergence/divergence.
//!
//! This is intended as a reusable foundation for elevation/bathymetry features
//! (trenches, arcs, ridges) that need to be anchored to plate boundaries rather
//! than to a diffused scalar field.

use std::collections::HashSet;

use glam::Vec3;

use super::dynamics::{Dynamics, PlateType};
use super::{Plates, Tessellation};

/// Kinematic boundary classification based on relative motion across the boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    /// Plates are moving toward each other across the boundary.
    Convergent,
    /// Plates are moving away from each other across the boundary.
    Divergent,
    /// Relative motion is mostly along the boundary (near-zero normal component).
    Transform,
}

/// Which side subducts for a convergent boundary edge.
///
/// This is expressed in terms of the `PlateBoundaryEdge` endpoints:
/// - `ASubducts`: side A is the subducting side, B is overriding.
/// - `BSubducts`: side B is the subducting side, A is overriding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubductionPolarity {
    ASubducts,
    BSubducts,
}

/// A single boundary edge between two adjacent cells on different plates.
///
/// This is stored once per adjacency pair (A < B by cell index).
#[derive(Clone, Copy, Debug)]
pub struct PlateBoundaryEdge {
    pub cell_a: usize,
    pub cell_b: usize,
    pub plate_a: usize,
    pub plate_b: usize,
    pub type_a: PlateType,
    pub type_b: PlateType,
    /// Boundary midpoint (unit vector on the sphere).
    pub boundary_point: Vec3,
    /// Arc length of the shared Voronoi edge (radians).
    pub edge_length: f32,
    /// Signed convergence across the boundary normal (positive = closing).
    pub convergence: f32,
    pub kind: BoundaryKind,
    /// Subduction polarity for convergent boundaries where it is defined.
    pub subduction: Option<SubductionPolarity>,
}

const CONVERGENCE_EPS: f32 = 1e-6;

fn classify_kind(convergence: f32) -> BoundaryKind {
    if convergence > CONVERGENCE_EPS {
        BoundaryKind::Convergent
    } else if convergence < -CONVERGENCE_EPS {
        BoundaryKind::Divergent
    } else {
        BoundaryKind::Transform
    }
}

fn subduction_polarity(
    kind: BoundaryKind,
    type_a: PlateType,
    type_b: PlateType,
    plate_a: usize,
    plate_b: usize,
) -> Option<SubductionPolarity> {
    if kind != BoundaryKind::Convergent {
        return None;
    }

    match (type_a, type_b) {
        // Ocean–continent: oceanic side subducts.
        (PlateType::Oceanic, PlateType::Continental) => Some(SubductionPolarity::ASubducts),
        (PlateType::Continental, PlateType::Oceanic) => Some(SubductionPolarity::BSubducts),
        // Ocean–ocean: pick a deterministic polarity for now (refine later using age proxy).
        (PlateType::Oceanic, PlateType::Oceanic) => {
            if plate_a <= plate_b {
                Some(SubductionPolarity::ASubducts)
            } else {
                Some(SubductionPolarity::BSubducts)
            }
        }
        // Continent–continent: collision; no subduction polarity.
        (PlateType::Continental, PlateType::Continental) => None,
    }
}

fn shared_edge_length(tessellation: &Tessellation, cell_a: usize, cell_b: usize) -> f32 {
    let verts_a: HashSet<usize> = tessellation.voronoi.cells[cell_a]
        .vertex_indices
        .iter()
        .copied()
        .collect();
    let verts_b: HashSet<usize> = tessellation.voronoi.cells[cell_b]
        .vertex_indices
        .iter()
        .copied()
        .collect();
    let shared: Vec<usize> = verts_a.intersection(&verts_b).copied().collect();

    if shared.len() == 2 {
        let v0 = tessellation.voronoi.vertices[shared[0]];
        let v1 = tessellation.voronoi.vertices[shared[1]];
        v0.dot(v1).clamp(-1.0, 1.0).acos()
    } else {
        // Fallback: adjacency without a clean shared edge (should be rare).
        0.1
    }
}

fn boundary_tangent_normal(boundary_point: Vec3, cell_pos: Vec3, neighbor_pos: Vec3) -> Vec3 {
    // Same construction as stress.rs: compute a stable tangent-space normal
    // pointing across the boundary from cell -> neighbor.
    let chord = neighbor_pos - cell_pos;
    let tangent_normal = chord - boundary_point * chord.dot(boundary_point);
    if tangent_normal.length_squared() > 1e-10 {
        tangent_normal.normalize()
    } else {
        let up = if boundary_point.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        boundary_point.cross(up).normalize()
    }
}

/// Collect all unique plate-boundary edges and classify them by convergence.
pub fn collect_plate_boundaries(
    tessellation: &Tessellation,
    plates: &Plates,
    dynamics: &Dynamics,
) -> Vec<PlateBoundaryEdge> {
    let n = tessellation.num_cells();
    let mut boundaries = Vec::new();

    for cell_a in 0..n {
        let plate_a = plates.cell_plate[cell_a] as usize;
        let type_a = dynamics.plate_type(plate_a);
        let pos_a = tessellation.cell_center(cell_a);

        for &cell_b in tessellation.neighbors(cell_a) {
            if cell_b <= cell_a {
                continue; // store each adjacency once
            }

            let plate_b = plates.cell_plate[cell_b] as usize;
            if plate_a == plate_b {
                continue;
            }

            let type_b = dynamics.plate_type(plate_b);
            let pos_b = tessellation.cell_center(cell_b);

            let boundary_point = (pos_a + pos_b).normalize();
            let vel_a = dynamics.euler_pole(plate_a).velocity_at(boundary_point);
            let vel_b = dynamics.euler_pole(plate_b).velocity_at(boundary_point);
            let relative_vel = vel_a - vel_b;

            let tangent_normal = boundary_tangent_normal(boundary_point, pos_a, pos_b);
            let convergence = relative_vel.dot(tangent_normal);

            let kind = classify_kind(convergence);
            let subduction = subduction_polarity(kind, type_a, type_b, plate_a, plate_b);
            let edge_length = shared_edge_length(tessellation, cell_a, cell_b);

            boundaries.push(PlateBoundaryEdge {
                cell_a,
                cell_b,
                plate_a,
                plate_b,
                type_a,
                type_b,
                boundary_point,
                edge_length,
                convergence,
                kind,
                subduction,
            });
        }
    }

    boundaries
}

