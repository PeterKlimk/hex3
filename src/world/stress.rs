//! Stress field calculation from plate interactions.

use super::boundary::{collect_plate_boundaries, BoundaryKind, PlateBoundaryEdge};
use super::constants::*;
use super::dynamics::{Dynamics, PlateType};
use super::{Plates, Tessellation};

/// Stress field from tectonic plate interactions.
pub struct StressField {
    /// Boundary stress at each cell (before propagation).
    pub boundary_stress: Vec<f32>,

    /// Propagated stress at each cell (positive = compression, negative = tension).
    pub cell_stress: Vec<f32>,
}

impl StressField {
    /// Calculate stress field from plate dynamics.
    pub fn calculate(tessellation: &Tessellation, plates: &Plates, dynamics: &Dynamics) -> Self {
        let boundaries = collect_plate_boundaries(tessellation, plates, dynamics);
        let boundary_stress =
            calculate_boundary_stress_from_edges(tessellation.num_cells(), &boundaries);

        let cell_stress =
            propagate_stress(&boundary_stress, plates, tessellation);

        Self {
            boundary_stress,
            cell_stress,
        }
    }
}

fn interaction_multiplier(kind: BoundaryKind, type_a: PlateType, type_b: PlateType) -> f32 {
    match kind {
        BoundaryKind::Convergent => match (type_a, type_b) {
            (PlateType::Continental, PlateType::Continental) => CONV_CONT_CONT,
            (PlateType::Oceanic, PlateType::Oceanic) => CONV_OCEAN_OCEAN,
            (PlateType::Continental, PlateType::Oceanic) => CONV_CONT_OCEAN,
            (PlateType::Oceanic, PlateType::Continental) => CONV_OCEAN_CONT,
        },
        BoundaryKind::Divergent | BoundaryKind::Transform => match (type_a, type_b) {
            (PlateType::Continental, PlateType::Continental) => DIV_CONT_CONT,
            (PlateType::Oceanic, PlateType::Oceanic) => DIV_OCEAN_OCEAN,
            (PlateType::Continental, PlateType::Oceanic) => DIV_CONT_OCEAN,
            (PlateType::Oceanic, PlateType::Continental) => DIV_OCEAN_CONT,
        },
    }
}

/// Calculate boundary stress for each cell from an explicit boundary edge list.
fn calculate_boundary_stress_from_edges(num_cells: usize, boundaries: &[PlateBoundaryEdge]) -> Vec<f32> {
    let mut stress = vec![0.0f32; num_cells];
    for b in boundaries {
        let mult_a = interaction_multiplier(b.kind, b.type_a, b.type_b);
        let mult_b = interaction_multiplier(b.kind, b.type_b, b.type_a);

        stress[b.cell_a] += b.convergence * mult_a * b.edge_length * STRESS_SCALE;
        stress[b.cell_b] += b.convergence * mult_b * b.edge_length * STRESS_SCALE;
    }

    stress
}

/// Propagate stress from boundary cells using screened diffusion.
fn propagate_stress(
    boundary_stress: &[f32],
    plates: &Plates,
    tessellation: &Tessellation,
) -> Vec<f32> {
    let num_cells = boundary_stress.len();
    let num_plates = plates.num_plates;

    // Compute mean neighbor distance to calibrate λ
    let mean_neighbor_dist = compute_mean_neighbor_distance(tessellation);

    // Convert decay length to λ
    let k = STRESS_DECAY_LENGTH / mean_neighbor_dist;
    let lambda = k * k;

    // Build plate membership lists
    let mut plate_cells: Vec<Vec<usize>> = vec![Vec::new(); num_plates];
    for (cell_idx, &plate) in plates.cell_plate.iter().enumerate() {
        plate_cells[plate as usize].push(cell_idx);
    }

    let mut stress = vec![0.0f32; num_cells];

    // Solve independently per plate using Gauss-Seidel iteration
    for (plate_id, cells) in plate_cells.iter().enumerate() {
        if cells.is_empty() {
            continue;
        }

        // Build local index mapping
        let mut global_to_local: Vec<usize> = vec![usize::MAX; num_cells];
        for (local_idx, &global_idx) in cells.iter().enumerate() {
            global_to_local[global_idx] = local_idx;
        }

        // Build plate-restricted adjacency and diagonal terms
        let mut local_neighbors: Vec<Vec<usize>> = Vec::with_capacity(cells.len());
        let mut diag: Vec<f32> = Vec::with_capacity(cells.len());

        for &global_idx in cells {
            let neighbors: Vec<usize> = tessellation
                .neighbors(global_idx)
                .iter()
                .filter(|&&n| plates.cell_plate[n] == plate_id as u32)
                .map(|&n| global_to_local[n])
                .collect();

            let degree = neighbors.len() as f32;
            diag.push(1.0 + lambda * degree);
            local_neighbors.push(neighbors);
        }

        // Initialize with boundary stress
        let mut s: Vec<f32> = cells.iter().map(|&i| boundary_stress[i]).collect();
        let b: Vec<f32> = cells.iter().map(|&i| boundary_stress[i]).collect();

        // Gauss-Seidel iteration
        let omega = DIFFUSION_DAMPING;

        for _ in 0..DIFFUSION_MAX_ITERS {
            let mut max_change: f32 = 0.0;

            for local_idx in 0..cells.len() {
                let neighbor_sum: f32 = local_neighbors[local_idx].iter().map(|&n| s[n]).sum();
                let candidate = (b[local_idx] + lambda * neighbor_sum) / diag[local_idx];
                let new_val = (1.0 - omega) * s[local_idx] + omega * candidate;

                max_change = max_change.max((new_val - s[local_idx]).abs());
                s[local_idx] = new_val;
            }

            if max_change < DIFFUSION_TOLERANCE {
                break;
            }
        }

        // Copy results back
        for (local_idx, &global_idx) in cells.iter().enumerate() {
            stress[global_idx] = s[local_idx];
        }
    }

    stress
}

/// Compute mean angular distance between neighboring cells.
fn compute_mean_neighbor_distance(tessellation: &Tessellation) -> f32 {
    let mut total_dist: f32 = 0.0;
    let mut count: usize = 0;

    for i in 0..tessellation.num_cells() {
        let pos_i = tessellation.cell_center(i);
        for &j in tessellation.neighbors(i) {
            if j > i {
                let pos_j = tessellation.cell_center(j);
                let dist = pos_i.dot(pos_j).clamp(-1.0, 1.0).acos();
                total_dist += dist;
                count += 1;
            }
        }
    }

    if count > 0 {
        total_dist / count as f32
    } else {
        0.03
    }
}
