//! Plate boundary extraction and classification.
//!
//! This module builds an explicit list of plate-boundary edges (between adjacent
//! tessellation cells with different plate assignments), and classifies each edge
//! by kinematic convergence/divergence.
//!
//! This is intended as a reusable foundation for elevation/bathymetry features
//! (trenches, arcs, ridges) that need to be anchored to plate boundaries rather
//! than to a diffused scalar field.

use std::collections::{HashMap, HashSet};

use glam::Vec3;

use super::constants::{TRANSFORM_NORMAL_THRESHOLD, TRANSFORM_RATIO};
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
    /// Signed shear along the boundary (tangential component of relative velocity).
    pub shear: f32,
    /// Magnitude of relative velocity (for activity calculations).
    pub relative_speed: f32,
    pub kind: BoundaryKind,
    /// Subduction polarity for convergent boundaries where it is defined.
    pub subduction: Option<SubductionPolarity>,
}

#[cfg(test)]
/// Classify boundary kind using both normal (convergence) and tangential (shear) components.
///
/// Transform boundaries require:
/// 1. Shear magnitude dominates convergence by TRANSFORM_RATIO
/// 2. Convergence magnitude is below TRANSFORM_NORMAL_THRESHOLD
fn classify_kind(convergence: f32, shear: f32) -> BoundaryKind {
    let normal_mag = convergence.abs();
    let tangent_mag = shear.abs();

    // Transform if tangential motion dominates and normal component is small
    if tangent_mag > normal_mag * TRANSFORM_RATIO && normal_mag < TRANSFORM_NORMAL_THRESHOLD {
        BoundaryKind::Transform
    } else if convergence > TRANSFORM_NORMAL_THRESHOLD {
        BoundaryKind::Convergent
    } else if convergence < -TRANSFORM_NORMAL_THRESHOLD {
        BoundaryKind::Divergent
    } else {
        // Low-normal-motion boundaries are treated as transform/inactive for feature generation.
        BoundaryKind::Transform
    }
}

fn shared_edge_length(tessellation: &Tessellation, cell_a: usize, cell_b: usize) -> f32 {
    let verts_a: HashSet<u32> = tessellation
        .voronoi
        .cell(cell_a)
        .vertex_indices
        .iter()
        .copied()
        .collect();
    let verts_b: HashSet<u32> = tessellation
        .voronoi
        .cell(cell_b)
        .vertex_indices
        .iter()
        .copied()
        .collect();
    let shared: Vec<u32> = verts_a.intersection(&verts_b).copied().collect();

    let shared_len = shared.len();
    if shared_len == 2 {
        let v0 = tessellation.voronoi.vertices[shared[0] as usize];
        let v1 = tessellation.voronoi.vertices[shared[1] as usize];
        v0.dot(v1).clamp(-1.0, 1.0).acos()
    } else {
        debug_assert_eq!(
            shared_len, 2,
            "adjacent cells {cell_a} and {cell_b} share {shared_len} vertices"
        );
        // Fallback: approximate the boundary edge length from the cell-center separation.
        // This should be unreachable for valid Voronoi topology.
        let pos_a = tessellation.cell_center(cell_a);
        let pos_b = tessellation.cell_center(cell_b);
        0.5 * pos_a.dot(pos_b).clamp(-1.0, 1.0).acos()
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

/// Canonical plate pair key: (min, max) for consistent lookup.
fn plate_pair_key(plate_a: usize, plate_b: usize) -> (usize, usize) {
    if plate_a <= plate_b {
        (plate_a, plate_b)
    } else {
        (plate_b, plate_a)
    }
}

#[derive(Default, Clone, Copy)]
struct PlatePairStats {
    total_edge_length: f32,
    sum_normal: f32,
    sum_abs_shear: f32,
    active_len_pos: f32,
    active_len_neg: f32,
}

fn classify_plate_pair(stats: PlatePairStats) -> BoundaryKind {
    use super::constants::{PLATE_PAIR_MIN_ACTIVE_LENGTH, PLATE_PAIR_MIN_BOUNDARY_LENGTH};

    if stats.total_edge_length < PLATE_PAIR_MIN_BOUNDARY_LENGTH {
        return BoundaryKind::Transform;
    }

    let mean_normal = stats.sum_normal / stats.total_edge_length;
    let mean_abs_shear = stats.sum_abs_shear / stats.total_edge_length;

    if mean_abs_shear > mean_normal.abs() * TRANSFORM_RATIO
        && mean_normal.abs() < TRANSFORM_NORMAL_THRESHOLD
    {
        return BoundaryKind::Transform;
    }

    if mean_normal > TRANSFORM_NORMAL_THRESHOLD
        && stats.active_len_pos >= PLATE_PAIR_MIN_ACTIVE_LENGTH
    {
        BoundaryKind::Convergent
    } else if mean_normal < -TRANSFORM_NORMAL_THRESHOLD
        && stats.active_len_neg >= PLATE_PAIR_MIN_ACTIVE_LENGTH
    {
        BoundaryKind::Divergent
    } else {
        BoundaryKind::Transform
    }
}

/// Compute stable subduction polarity for ocean-ocean plate pairs.
///
/// For ocean-ocean convergence, we need a consistent polarity across the entire
/// boundary between two plates. This function aggregates edge-weighted convergence
/// to determine which side "wins" as the subductor.
fn compute_ocean_ocean_polarities(
    tessellation: &Tessellation,
    plates: &Plates,
    dynamics: &Dynamics,
) -> HashMap<(usize, usize), SubductionPolarity> {
    let n = tessellation.num_cells();

    // Accumulate weighted votes per plate pair
    // Key: (min_plate, max_plate), Value: (votes_for_min_subducts, votes_for_max_subducts)
    let mut votes: HashMap<(usize, usize), (f32, f32)> = HashMap::new();

    for cell_a in 0..n {
        let plate_a = plates.cell_plate[cell_a] as usize;
        let type_a = dynamics.plate_type(plate_a);

        if type_a != PlateType::Oceanic {
            continue;
        }

        let pos_a = tessellation.cell_center(cell_a);

        for &cell_b in tessellation.neighbors(cell_a) {
            if cell_b <= cell_a {
                continue;
            }

            let plate_b = plates.cell_plate[cell_b] as usize;
            if plate_a == plate_b {
                continue;
            }

            let type_b = dynamics.plate_type(plate_b);
            if type_b != PlateType::Oceanic {
                continue;
            }

            let pos_b = tessellation.cell_center(cell_b);
            let boundary_point = (pos_a + pos_b).normalize();

            let vel_a = dynamics.euler_pole(plate_a).velocity_at(boundary_point);
            let vel_b = dynamics.euler_pole(plate_b).velocity_at(boundary_point);
            let relative_vel = vel_a - vel_b;

            let tangent_normal = boundary_tangent_normal(boundary_point, pos_a, pos_b);
            let convergence = relative_vel.dot(tangent_normal);

            // Only care about convergent edges for subduction
            if convergence <= TRANSFORM_NORMAL_THRESHOLD {
                continue;
            }

            let edge_length = shared_edge_length(tessellation, cell_a, cell_b);
            let weight = convergence * edge_length;

            let key = plate_pair_key(plate_a, plate_b);
            let entry = votes.entry(key).or_insert((0.0, 0.0));

            // Vote for which plate subducts based on each plate's *toward-boundary* component.
            //
            // Clamp "toward" magnitudes to 0 to avoid negative values (motion away from the
            // boundary) skewing polarity decisions on mixed-motion segments.
            let toward_a_to_b = vel_a.dot(tangent_normal).max(0.0);
            let toward_b_to_a = vel_b.dot(-tangent_normal).max(0.0);

            // Compare min-plate vs max-plate toward components for the canonical (min,max) pair.
            let (min_toward, max_toward) = if plate_a < plate_b {
                (toward_a_to_b, toward_b_to_a)
            } else {
                (toward_b_to_a, toward_a_to_b)
            };

            if (min_toward - max_toward).abs() < 1e-6 {
                // Tie: split the vote to avoid systematic bias.
                entry.0 += 0.5 * weight;
                entry.1 += 0.5 * weight;
            } else if min_toward > max_toward {
                entry.0 += weight; // min plate subducts
            } else {
                entry.1 += weight; // max plate subducts
            }
        }
    }

    // Convert votes to polarities
    votes
        .into_iter()
        .map(|(key, (min_votes, max_votes))| {
            let polarity = if min_votes >= max_votes {
                // min plate subducts
                SubductionPolarity::ASubducts
            } else {
                SubductionPolarity::BSubducts
            };
            (key, polarity)
        })
        .collect()
}

/// Look up subduction polarity, using pre-computed ocean-ocean polarities for stability.
fn lookup_subduction_polarity(
    kind: BoundaryKind,
    type_a: PlateType,
    type_b: PlateType,
    plate_a: usize,
    plate_b: usize,
    ocean_ocean_polarities: &HashMap<(usize, usize), SubductionPolarity>,
) -> Option<SubductionPolarity> {
    if kind != BoundaryKind::Convergent {
        return None;
    }

    match (type_a, type_b) {
        // Ocean–continent: oceanic side subducts (deterministic)
        (PlateType::Oceanic, PlateType::Continental) => Some(SubductionPolarity::ASubducts),
        (PlateType::Continental, PlateType::Oceanic) => Some(SubductionPolarity::BSubducts),

        // Ocean–ocean: use pre-computed stable polarity
        (PlateType::Oceanic, PlateType::Oceanic) => {
            let key = plate_pair_key(plate_a, plate_b);
            let base_polarity = ocean_ocean_polarities
                .get(&key)
                .copied()
                .unwrap_or(SubductionPolarity::ASubducts);

            // Adjust polarity based on which plate is A vs B in this edge
            if plate_a <= plate_b {
                Some(base_polarity)
            } else {
                // Flip polarity since A and B are swapped relative to the key
                Some(match base_polarity {
                    SubductionPolarity::ASubducts => SubductionPolarity::BSubducts,
                    SubductionPolarity::BSubducts => SubductionPolarity::ASubducts,
                })
            }
        }

        // Continent–continent: collision, no subduction
        (PlateType::Continental, PlateType::Continental) => None,
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
    let mut pair_stats: HashMap<(usize, usize), PlatePairStats> = HashMap::new();

    // Pass 1: compute kinematics for each boundary edge and aggregate per plate pair.
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

            // Compute normal (convergence) and tangential (shear) components
            let tangent_normal = boundary_tangent_normal(boundary_point, pos_a, pos_b);
            let tangent_along = boundary_point.cross(tangent_normal).normalize();

            let convergence = relative_vel.dot(tangent_normal);
            let shear = relative_vel.dot(tangent_along);
            let relative_speed = relative_vel.length();

            let edge_length = shared_edge_length(tessellation, cell_a, cell_b);

            let key = plate_pair_key(plate_a, plate_b);
            let entry = pair_stats.entry(key).or_default();
            entry.total_edge_length += edge_length;
            entry.sum_normal += convergence * edge_length;
            entry.sum_abs_shear += shear.abs() * edge_length;
            if convergence >= TRANSFORM_NORMAL_THRESHOLD {
                entry.active_len_pos += edge_length;
            } else if convergence <= -TRANSFORM_NORMAL_THRESHOLD {
                entry.active_len_neg += edge_length;
            }

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
                shear,
                relative_speed,
                kind: BoundaryKind::Transform, // filled in pass 2
                subduction: None,              // filled in pass 2
            });
        }
    }

    // Pass 2: classify each touching plate pair and apply to edges (regime is coherent
    // at plate-pair scale; local sign noise only modulates strength).
    let pair_kind: HashMap<(usize, usize), BoundaryKind> = pair_stats
        .into_iter()
        .map(|(k, stats)| (k, classify_plate_pair(stats)))
        .collect();

    // Pre-compute stable ocean-ocean subduction polarities (used only for convergent edges).
    let ocean_ocean_polarities = compute_ocean_ocean_polarities(tessellation, plates, dynamics);

    for b in &mut boundaries {
        let key = plate_pair_key(b.plate_a, b.plate_b);
        let kind = pair_kind
            .get(&key)
            .copied()
            .unwrap_or(BoundaryKind::Transform);
        b.kind = kind;
        b.subduction = lookup_subduction_polarity(
            kind,
            b.type_a,
            b.type_b,
            b.plate_a,
            b.plate_b,
            &ocean_ocean_polarities,
        );
    }

    boundaries
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::world::Tessellation;

    #[test]
    fn classify_kind_requires_meaningful_normal_motion() {
        let t = TRANSFORM_NORMAL_THRESHOLD;

        assert_eq!(classify_kind(0.5 * t, 0.0), BoundaryKind::Transform);
        assert_eq!(classify_kind(-0.5 * t, 0.0), BoundaryKind::Transform);
        assert_eq!(classify_kind(1.1 * t, 0.0), BoundaryKind::Convergent);
        assert_eq!(classify_kind(-1.1 * t, 0.0), BoundaryKind::Divergent);

        // Strong shear with small normal should be transform.
        assert_eq!(classify_kind(0.5 * t, 10.0 * t), BoundaryKind::Transform);
    }

    #[test]
    fn plate_pair_classification_gates_short_or_weak_contacts() {
        use super::super::constants::{
            PLATE_PAIR_MIN_ACTIVE_LENGTH, PLATE_PAIR_MIN_BOUNDARY_LENGTH,
        };

        // Too short: always transform.
        let mut stats = PlatePairStats::default();
        stats.total_edge_length = PLATE_PAIR_MIN_BOUNDARY_LENGTH * 0.5;
        stats.sum_normal = stats.total_edge_length * 10.0 * TRANSFORM_NORMAL_THRESHOLD;
        stats.active_len_pos = stats.total_edge_length;
        assert_eq!(classify_plate_pair(stats), BoundaryKind::Transform);

        // Long enough but with too little consistent-sign activity: transform.
        let mut stats = PlatePairStats::default();
        stats.total_edge_length = PLATE_PAIR_MIN_BOUNDARY_LENGTH * 1.5;
        stats.sum_normal = stats.total_edge_length * 2.0 * TRANSFORM_NORMAL_THRESHOLD;
        stats.active_len_pos = PLATE_PAIR_MIN_ACTIVE_LENGTH * 0.5;
        assert_eq!(classify_plate_pair(stats), BoundaryKind::Transform);

        // Coherent convergent: convergent.
        let mut stats = PlatePairStats::default();
        stats.total_edge_length = PLATE_PAIR_MIN_BOUNDARY_LENGTH * 1.5;
        stats.sum_normal = stats.total_edge_length * 2.0 * TRANSFORM_NORMAL_THRESHOLD;
        stats.active_len_pos = PLATE_PAIR_MIN_ACTIVE_LENGTH * 1.5;
        assert_eq!(classify_plate_pair(stats), BoundaryKind::Convergent);
    }

    #[test]
    fn adjacent_cells_share_exactly_two_vertices() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let tessellation = Tessellation::generate(400, 1, &mut rng);
        let voronoi = &tessellation.voronoi;

        for a in 0..tessellation.num_cells() {
            for &b in tessellation.neighbors(a) {
                if b <= a {
                    continue;
                }

                let verts_a: HashSet<usize> =
                    voronoi.cell(a).vertex_indices.iter().copied().collect();
                let verts_b: HashSet<usize> =
                    voronoi.cell(b).vertex_indices.iter().copied().collect();
                let shared = verts_a.intersection(&verts_b).count();

                assert_eq!(shared, 2, "cells {a} and {b} share {shared} vertices");
            }
        }
    }
}
