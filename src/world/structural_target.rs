//! Frozen legacy target-domain recovery for Structural Mountain V0.
//!
//! Selection is anchor-first and uses only the unchanged legacy Stage-4 surface.
//! It does not rerun a mutable dossier rank and does not write terrain.

use std::collections::VecDeque;

use glam::Vec3;

use super::{elevation_to_km, Hydrology, Tessellation, PLANET_RADIUS_KM};

pub const FIXED_STRUCTURAL_TARGET_ANCHOR: [f32; 3] = [0.604_890_4, 0.641_965_5, 0.471_156_1];
pub const STRUCTURAL_TARGET_THRESHOLD_KM: f32 = 1.5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralTargetError {
    LengthMismatch,
    InvalidAnchor,
    InvalidThreshold,
    AnchorBelowThreshold { cell: usize },
    InvalidReceiver { cell: usize, receiver: usize },
    DrainageCycle,
}

impl std::fmt::Display for StructuralTargetError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for StructuralTargetError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralTargetProvenance {
    pub core_integration_cut_cells: usize,
    pub buffer_integration_cut_cells: usize,
    pub core_breached_source_cells: usize,
    pub buffer_breached_source_cells: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralTargetDomain {
    pub anchor: Vec3,
    pub anchor_cell: usize,
    pub peak_cell: usize,
    pub threshold_km: f32,
    pub core_cells: Vec<usize>,
    /// Non-core land in catchments whose terminal outlet also receives core runoff.
    pub catchment_buffer_cells: Vec<usize>,
    /// Parallel to `catchment_buffer_cells`.
    pub buffer_terminal_cells: Vec<usize>,
    pub core_area_km2: f64,
    pub catchment_buffer_area_km2: f64,
    pub provenance: StructuralTargetProvenance,
}

impl StructuralTargetDomain {
    pub fn all_cells(&self) -> Vec<usize> {
        let mut cells = self.core_cells.clone();
        cells.extend_from_slice(&self.catchment_buffer_cells);
        cells.sort_unstable();
        cells
    }
}

pub fn select_fixed_structural_target(
    tessellation: &Tessellation,
    final_elevation: &[f32],
    hydrology: &Hydrology,
) -> Result<StructuralTargetDomain, StructuralTargetError> {
    select_structural_target(
        tessellation,
        final_elevation,
        hydrology,
        FIXED_STRUCTURAL_TARGET_ANCHOR,
        STRUCTURAL_TARGET_THRESHOLD_KM,
    )
}

pub fn select_structural_target(
    tessellation: &Tessellation,
    final_elevation: &[f32],
    hydrology: &Hydrology,
    anchor_xyz: [f32; 3],
    threshold_km: f32,
) -> Result<StructuralTargetDomain, StructuralTargetError> {
    let n = tessellation.num_cells();
    if final_elevation.len() != n
        || hydrology.drainage_dir.len() != n
        || hydrology.integration_breached_source.len() != n
    {
        return Err(StructuralTargetError::LengthMismatch);
    }
    let anchor = Vec3::from_array(anchor_xyz);
    if !anchor.is_finite() || anchor.length_squared() <= 1e-12 {
        return Err(StructuralTargetError::InvalidAnchor);
    }
    if !threshold_km.is_finite() {
        return Err(StructuralTargetError::InvalidThreshold);
    }
    let anchor = anchor.normalize();
    let anchor_cell = (0..n)
        .min_by(|&a, &b| {
            tessellation
                .cell_center(a)
                .distance_squared(anchor)
                .total_cmp(&tessellation.cell_center(b).distance_squared(anchor))
                .then_with(|| a.cmp(&b))
        })
        .ok_or(StructuralTargetError::LengthMismatch)?;
    if elevation_to_km(final_elevation[anchor_cell]) < threshold_km {
        return Err(StructuralTargetError::AnchorBelowThreshold { cell: anchor_cell });
    }

    let mut core_mask = vec![false; n];
    let mut queue = VecDeque::from([anchor_cell]);
    core_mask[anchor_cell] = true;
    let mut core_cells = Vec::new();
    while let Some(cell) = queue.pop_front() {
        core_cells.push(cell);
        for &neighbor in tessellation.neighbors(cell) {
            if !core_mask[neighbor] && elevation_to_km(final_elevation[neighbor]) >= threshold_km {
                core_mask[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    core_cells.sort_unstable();
    let peak_cell = *core_cells
        .iter()
        .max_by(|&&a, &&b| {
            final_elevation[a]
                .total_cmp(&final_elevation[b])
                .then_with(|| b.cmp(&a))
        })
        .expect("anchor creates a nonempty component");

    let terminal_cells = drainage_terminal_cells(&hydrology.drainage_dir)?;
    let core_terminals: std::collections::BTreeSet<_> = core_cells
        .iter()
        .map(|&cell| terminal_cells[cell])
        .collect();
    let mut catchment_buffer_cells = Vec::new();
    let mut buffer_terminal_cells = Vec::new();
    for (cell, &terminal) in terminal_cells.iter().enumerate() {
        if !core_mask[cell] && !hydrology.is_submerged(cell) && core_terminals.contains(&terminal) {
            catchment_buffer_cells.push(cell);
            buffer_terminal_cells.push(terminal);
        }
    }

    let areas = tessellation.cell_areas();
    let area_scale = f64::from(PLANET_RADIUS_KM).powi(2);
    let area = |cells: &[usize]| {
        cells
            .iter()
            .map(|&cell| f64::from(areas[cell]) * area_scale)
            .sum::<f64>()
    };
    let count = |cells: &[usize], predicate: &dyn Fn(usize) -> bool| {
        cells.iter().filter(|&&cell| predicate(cell)).count()
    };
    let core_integration_cut_cells = count(&core_cells, &|cell| {
        hydrology.was_lowered_by_integration(cell)
    });
    let buffer_integration_cut_cells = count(&catchment_buffer_cells, &|cell| {
        hydrology.was_lowered_by_integration(cell)
    });
    let core_breached_source_cells = count(&core_cells, &|cell| {
        hydrology.integration_breached_source[cell]
    });
    let buffer_breached_source_cells = count(&catchment_buffer_cells, &|cell| {
        hydrology.integration_breached_source[cell]
    });

    Ok(StructuralTargetDomain {
        anchor,
        anchor_cell,
        peak_cell,
        threshold_km,
        core_area_km2: area(&core_cells),
        catchment_buffer_area_km2: area(&catchment_buffer_cells),
        core_cells,
        catchment_buffer_cells,
        buffer_terminal_cells,
        provenance: StructuralTargetProvenance {
            core_integration_cut_cells,
            buffer_integration_cut_cells,
            core_breached_source_cells,
            buffer_breached_source_cells,
        },
    })
}

fn drainage_terminal_cells(
    drainage_dir: &[Option<usize>],
) -> Result<Vec<usize>, StructuralTargetError> {
    let n = drainage_dir.len();
    let mut indegree = vec![0usize; n];
    for (cell, receiver) in drainage_dir.iter().copied().enumerate() {
        if let Some(receiver) = receiver {
            if receiver >= n || receiver == cell {
                return Err(StructuralTargetError::InvalidReceiver { cell, receiver });
            }
            indegree[receiver] += 1;
        }
    }

    // Validate the whole supplied drainage graph, not only the selected branch.
    let mut topo = VecDeque::new();
    for (cell, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            topo.push_back(cell);
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut processed = 0;
    while let Some(cell) = topo.pop_front() {
        processed += 1;
        order.push(cell);
        if let Some(receiver) = drainage_dir[cell] {
            indegree[receiver] -= 1;
            if indegree[receiver] == 0 {
                topo.push_back(receiver);
            }
        }
    }
    if processed != n {
        return Err(StructuralTargetError::DrainageCycle);
    }
    let mut terminals: Vec<_> = (0..n).collect();
    for cell in order.into_iter().rev() {
        if let Some(receiver) = drainage_dir[cell] {
            terminals[cell] = terminals[receiver];
        }
    }
    Ok(terminals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_catchments_include_other_tributaries_and_exclude_other_outlets() {
        // 0 -> 1 -> 2(core) -> 3 sink; 4 -> 3 joins below the core; 5 is unrelated.
        let drainage = vec![Some(1), Some(2), Some(3), None, Some(3), None];
        assert_eq!(
            drainage_terminal_cells(&drainage).unwrap(),
            vec![3, 3, 3, 3, 3, 5]
        );
    }

    #[test]
    fn invalid_receiver_and_cycle_are_typed() {
        assert_eq!(
            drainage_terminal_cells(&[Some(2), None]),
            Err(StructuralTargetError::InvalidReceiver {
                cell: 0,
                receiver: 2
            })
        );
        assert_eq!(
            drainage_terminal_cells(&[Some(1), Some(0)]),
            Err(StructuralTargetError::DrainageCycle)
        );
    }
}
