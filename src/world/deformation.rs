//! Coarse thin-sheet crustal deformation.
//!
//! T0 solves a screened horizontal deformation velocity relative to each rigid
//! plate, then updates crust thickness from conservative face fluxes. This is
//! deliberately a velocity/continuity system—not a height kernel. Later rungs
//! can replace the linear viscosity with viscoplastic rheology without changing
//! the state or conservation law.

use glam::Vec3;

use super::boundary::{PlateBoundaryEdge, SubductionPolarity};
use super::constants::*;
use super::crust::{Crust, CrustType};
use super::{Plates, Tessellation};

pub(crate) struct ThinSheetFields {
    /// Signed thickness change. Area-weighted integral equals retained magma;
    /// collision shortening itself only redistributes existing crust.
    pub thickness_delta: Vec<f32>,
    /// Accumulated absolute normal strain over the represented episode.
    pub strain: Vec<f32>,
    /// Tangent axis of strongest compression; zero where unresolved.
    pub compression_axis: Vec<Vec3>,
}

#[derive(Clone, Copy)]
struct SheetEdge {
    a: usize,
    b: usize,
    conductance: f32,
    face_length: f32,
    normal_a_to_b: Vec3,
}

/// Gravitational yield relaxation of a nonnegative thickness field (the
/// legacy-yield orogen rung, and the strength/gravity half of the thin-sheet T1
/// rheology): material above `yield_value` is mobile and spreads by screened
/// diffusion; material below is statically supported and NEVER moves. Sub-yield
/// terrain is therefore untouched BY CONSTRUCTION — the operator can only cap
/// over-strength loads and build their flanks, not smooth ordinary belts.
/// Conserves area-weighted volume (up to the nonnegativity clamp). Edges are
/// ungated: gravitational crust flow ignores plate sutures and coastlines
/// (Tibet spreads regardless of whose plate the foreland is).
pub(crate) fn yield_relax(
    tess: &Tessellation,
    field: &[f32],
    // Per-cell strength threshold (width-aware yield: wide belts support taller
    // loads — Earth's peak-vs-belt-width curve; a uniform slice reproduces the
    // scalar rung).
    yield_value: &[f32],
    tau: f32,
    picard_steps: usize,
) -> Vec<f32> {
    let n = tess.num_cells();
    if tau <= 0.0
        || picard_steps == 0
        || !field
            .iter()
            .zip(yield_value.iter())
            .any(|(&h, &y)| h > y)
    {
        return field.to_vec();
    }
    let areas = tess.cell_areas();
    let mut edges = Vec::new();
    for i in 0..n {
        for &j in tess.neighbors(i) {
            if j <= i {
                continue;
            }
            let a = tess.cell_center(i);
            let b = tess.cell_center(j);
            let center_distance = (b - a).length();
            if center_distance <= 1e-8 {
                continue;
            }
            let face_length = tess.shared_edge_length(i, j);
            if face_length <= 0.0 {
                continue;
            }
            edges.push(SheetEdge {
                a: i,
                b: j,
                conductance: face_length / center_distance,
                face_length,
                normal_a_to_b: Vec3::ZERO, // unused by the scalar solve
            });
        }
    }
    yield_relax_on_edges(&areas, &edges, field, yield_value, tau, picard_steps)
}

fn yield_relax_on_edges(
    areas: &[f32],
    edges: &[SheetEdge],
    field: &[f32],
    yield_value: &[f32],
    tau: f32,
    picard_steps: usize,
) -> Vec<f32> {
    let n = field.len();
    let mut thickness = field.to_vec();
    for _ in 0..picard_steps {
        let mobile: Vec<f32> = thickness
            .iter()
            .zip(yield_value.iter())
            .map(|(&h, &y)| (h - y).max(0.0))
            .collect();
        if !mobile.iter().any(|&v| v > 0.0) {
            break;
        }
        let immobile: Vec<f32> = thickness
            .iter()
            .zip(mobile.iter())
            .map(|(&h, &m)| h - m)
            .collect();
        let relaxed = solve_screened_scalar(areas, edges, &mobile, tau / picard_steps as f32);
        for i in 0..n {
            thickness[i] = immobile[i] + relaxed[i].max(0.0);
        }
    }
    thickness
}

pub(crate) fn solve_thin_sheet(
    tess: &Tessellation,
    plates: &Plates,
    crust: &Crust,
    boundaries: &[PlateBoundaryEdge],
) -> ThinSheetFields {
    let n = tess.num_cells();
    let areas = tess.cell_areas();
    let mut target_sum = vec![Vec3::ZERO; n];
    let mut target_weight = vec![0.0f32; n];
    let mut magmatic_volume_rate = vec![0.0f32; n];

    for boundary in boundaries {
        let closing = boundary.convergence.max(0.0);
        if closing < TRANSFORM_NORMAL_THRESHOLD {
            continue;
        }
        let add_target =
            |cell: usize, magnitude: f32, target_sum: &mut [Vec3], target_weight: &mut [f32]| {
                if magnitude <= 0.0 {
                    return;
                }
                let center = tess.cell_center(cell);
                let toward_boundary =
                    boundary.boundary_point - center * center.dot(boundary.boundary_point);
                if toward_boundary.length_squared() <= 1e-12 {
                    return;
                }
                // Relative to rigid plate motion, collision resistance slows the
                // boundaryward edge: the anomaly points into the plate interior.
                let away = -toward_boundary.normalize();
                let weight = boundary.edge_length;
                target_sum[cell] += away * magnitude * weight;
                target_weight[cell] += weight;
            };

        match boundary.subduction {
            None if boundary.type_a == CrustType::Continental
                && boundary.type_b == CrustType::Continental =>
            {
                add_target(
                    boundary.cell_a,
                    0.5 * closing,
                    &mut target_sum,
                    &mut target_weight,
                );
                add_target(
                    boundary.cell_b,
                    0.5 * closing,
                    &mut target_sum,
                    &mut target_weight,
                );
            }
            Some(SubductionPolarity::ASubducts) => {
                let cell = boundary.cell_b;
                if boundary.type_b == CrustType::Continental {
                    add_target(
                        cell,
                        SUBDUCTION_COMPRESSION_COUPLING * closing,
                        &mut target_sum,
                        &mut target_weight,
                    );
                }
                magmatic_volume_rate[cell] += closing
                    * boundary.edge_length
                    * SUBDUCTION_MAGMATIC_ACCRETION
                    * CRUST_THICKNESS_OCEANIC;
            }
            Some(SubductionPolarity::BSubducts) => {
                let cell = boundary.cell_a;
                if boundary.type_a == CrustType::Continental {
                    add_target(
                        cell,
                        SUBDUCTION_COMPRESSION_COUPLING * closing,
                        &mut target_sum,
                        &mut target_weight,
                    );
                }
                magmatic_volume_rate[cell] += closing
                    * boundary.edge_length
                    * SUBDUCTION_MAGMATIC_ACCRETION
                    * CRUST_THICKNESS_OCEANIC;
            }
            _ => {}
        }
    }

    let target: Vec<Vec3> = target_sum
        .iter()
        .zip(target_weight.iter())
        .map(|(&sum, &weight)| {
            if weight > 0.0 {
                sum / weight
            } else {
                Vec3::ZERO
            }
        })
        .collect();

    // T0 treats each plate/crust block as a viscous sheet coupled to its rigid
    // plate interior. Cross-suture traction is represented by the boundary
    // target above; material flux itself remains within a motion unit.
    let mut edges = Vec::with_capacity(tess.adjacency.total_neighbor_entries() / 2);
    for i in 0..n {
        for &j in tess.neighbors(i) {
            if j <= i
                || plates.cell_plate[i] != plates.cell_plate[j]
                || crust.crust_type(i) != crust.crust_type(j)
            {
                continue;
            }
            let a = tess.cell_center(i);
            let b = tess.cell_center(j);
            let center_distance = (b - a).length();
            if center_distance <= 1e-8 {
                continue;
            }
            let face_length = tess.shared_edge_length(i, j);
            if face_length <= 0.0 {
                continue;
            }
            let midpoint = (a + b).normalize_or_zero();
            let chord = b - a;
            let normal = (chord - midpoint * midpoint.dot(chord)).normalize_or_zero();
            if normal == Vec3::ZERO {
                continue;
            }
            edges.push(SheetEdge {
                a: i,
                b: j,
                conductance: face_length / center_distance,
                face_length,
                normal_a_to_b: normal,
            });
        }
    }

    let velocity = solve_velocity(
        tess,
        &areas,
        &edges,
        &target,
        THIN_SHEET_VISCOSITY_DRAG_RATIO,
    );
    let reference_thickness: Vec<f32> = (0..n)
        .map(|i| match crust.crust_type(i) {
            CrustType::Continental => CRUST_THICKNESS_CONTINENTAL,
            CrustType::Oceanic => CRUST_THICKNESS_OCEANIC,
        })
        .collect();
    let mut strain_sum = vec![0.0f32; n];
    let mut strain_weight = vec![0.0f32; n];
    let mut strongest_compression = vec![0.0f32; n];
    let mut compression_axis = vec![Vec3::ZERO; n];

    for edge in &edges {
        let midpoint = (tess.cell_center(edge.a) + tess.cell_center(edge.b)).normalize_or_zero();
        let va = velocity[edge.a] - midpoint * midpoint.dot(velocity[edge.a]);
        let vb = velocity[edge.b] - midpoint * midpoint.dot(velocity[edge.b]);
        let center_distance = (tess.cell_center(edge.b) - tess.cell_center(edge.a))
            .length()
            .max(1e-8);
        let normal_strain = (vb - va).dot(edge.normal_a_to_b) / center_distance;
        let weight = edge.face_length;
        strain_sum[edge.a] += normal_strain.abs() * weight;
        strain_sum[edge.b] += normal_strain.abs() * weight;
        strain_weight[edge.a] += weight;
        strain_weight[edge.b] += weight;
        if normal_strain < -strongest_compression[edge.a] {
            strongest_compression[edge.a] = -normal_strain;
            compression_axis[edge.a] = edge.normal_a_to_b;
        }
        if normal_strain < -strongest_compression[edge.b] {
            strongest_compression[edge.b] = -normal_strain;
            compression_axis[edge.b] = edge.normal_a_to_b;
        }
    }

    let duration = THIN_SHEET_TECTONIC_TIME.max(0.0);
    // Retained magma is an actual positive material source. Spread its RATE
    // through the same sheet operator before integrating continuity.
    let magmatic_rate_source: Vec<f32> = magmatic_volume_rate
        .iter()
        .zip(areas.iter())
        .map(|(&rate, &area)| rate / area.max(1e-12))
        .collect();
    let magmatic_rate = solve_screened_scalar(
        &areas,
        &edges,
        &magmatic_rate_source,
        THIN_SHEET_VISCOSITY_DRAG_RATIO,
    );

    // Conservative upwind integration of dH/dt + div(Hu) = S_magma.
    // Substep count is derived from the graph CFL, never exposed as a terrain
    // knob. Upwinding preserves positivity without per-cell clamps or a global
    // episode rescale.
    let mut outgoing_fraction_rate = vec![0.0f32; n];
    for edge in &edges {
        let midpoint = (tess.cell_center(edge.a) + tess.cell_center(edge.b)).normalize_or_zero();
        let va = velocity[edge.a] - midpoint * midpoint.dot(velocity[edge.a]);
        let vb = velocity[edge.b] - midpoint * midpoint.dot(velocity[edge.b]);
        let normal_speed = (0.5 * (va + vb)).dot(edge.normal_a_to_b);
        if normal_speed > 0.0 {
            outgoing_fraction_rate[edge.a] +=
                normal_speed * edge.face_length / areas[edge.a].max(1e-12);
        } else {
            outgoing_fraction_rate[edge.b] +=
                -normal_speed * edge.face_length / areas[edge.b].max(1e-12);
        }
    }
    let max_courant = duration
        * outgoing_fraction_rate
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
    let substeps = (max_courant / 0.20).ceil().max(1.0) as usize;
    let dt = if substeps > 0 {
        duration / substeps as f32
    } else {
        0.0
    };
    let mut thickness = reference_thickness.clone();
    let mut volume_change = vec![0.0f32; n];
    for _ in 0..substeps {
        volume_change.fill(0.0);
        for edge in &edges {
            let midpoint =
                (tess.cell_center(edge.a) + tess.cell_center(edge.b)).normalize_or_zero();
            let va = velocity[edge.a] - midpoint * midpoint.dot(velocity[edge.a]);
            let vb = velocity[edge.b] - midpoint * midpoint.dot(velocity[edge.b]);
            let normal_speed = (0.5 * (va + vb)).dot(edge.normal_a_to_b);
            let donor_h = if normal_speed >= 0.0 {
                thickness[edge.a]
            } else {
                thickness[edge.b]
            };
            let flux = donor_h * normal_speed * edge.face_length;
            volume_change[edge.a] -= dt * flux;
            volume_change[edge.b] += dt * flux;
        }
        for i in 0..n {
            thickness[i] += volume_change[i] / areas[i].max(1e-12);
            thickness[i] += dt * magmatic_rate[i].max(0.0);
        }
    }
    let gravity_tau = THIN_SHEET_GRAVITATIONAL_DIFFUSIVITY.max(0.0) * duration;
    if gravity_tau > 0.0 {
        // Viscoplastic gravitational collapse: only thickness above the
        // strength-supported column is mobile. Picard substeps recompute the
        // yielded set while conserving mobile material exactly.
        const YIELD_PICARD_STEPS: usize = 4;
        for _ in 0..YIELD_PICARD_STEPS {
            let mobile: Vec<f32> = thickness
                .iter()
                .zip(reference_thickness.iter())
                .map(|(&height, &reference)| {
                    (height - reference - THIN_SHEET_YIELD_EXCESS).max(0.0)
                })
                .collect();
            if !mobile.iter().any(|&value| value > 0.0) {
                break;
            }
            let immobile: Vec<f32> = thickness
                .iter()
                .zip(mobile.iter())
                .map(|(&height, &yielded)| height - yielded)
                .collect();
            let relaxed = solve_screened_scalar(
                &areas,
                &edges,
                &mobile,
                gravity_tau / YIELD_PICARD_STEPS as f32,
            );
            for i in 0..n {
                thickness[i] = immobile[i] + relaxed[i].max(0.0);
            }
        }
    }
    let thickness_delta: Vec<f32> = thickness
        .iter()
        .zip(reference_thickness.iter())
        .map(|(&after, &before)| after - before)
        .collect();

    let strain: Vec<f32> = strain_sum
        .iter()
        .zip(strain_weight.iter())
        .map(|(&sum, &weight)| {
            if weight > 0.0 {
                duration * sum / weight
            } else {
                0.0
            }
        })
        .collect();

    log::debug!(
        "thin sheet: edges={}, substeps={}, max_courant={:.2}, min_H={:.3}, max_strain={:.3}, max_dH={:.3}",
        edges.len(),
        substeps,
        max_courant,
        thickness.iter().copied().fold(f32::INFINITY, f32::min),
        strain.iter().copied().fold(0.0f32, f32::max),
        thickness_delta
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
    );

    ThinSheetFields {
        thickness_delta,
        strain,
        compression_axis,
    }
}

fn solve_velocity(
    tess: &Tessellation,
    areas: &[f32],
    edges: &[SheetEdge],
    target: &[Vec3],
    coupling: f32,
) -> Vec<Vec3> {
    let xs: Vec<f32> = target.iter().map(|v| v.x).collect();
    let ys: Vec<f32> = target.iter().map(|v| v.y).collect();
    let zs: Vec<f32> = target.iter().map(|v| v.z).collect();
    let x = solve_screened_scalar(areas, edges, &xs, coupling);
    let y = solve_screened_scalar(areas, edges, &ys, coupling);
    let z = solve_screened_scalar(areas, edges, &zs, coupling);
    (0..target.len())
        .map(|i| {
            let center = tess.cell_center(i);
            let velocity = Vec3::new(x[i], y[i], z[i]);
            velocity - center * center.dot(velocity)
        })
        .collect()
}

fn solve_screened_scalar(
    areas: &[f32],
    edges: &[SheetEdge],
    source: &[f32],
    coupling: f32,
) -> Vec<f32> {
    if coupling <= 0.0 {
        return source.to_vec();
    }
    let n = source.len();
    let apply = |x: &[f32], out: &mut [f32]| {
        for i in 0..n {
            out[i] = areas[i] * x[i];
        }
        for edge in edges {
            let flux = coupling * edge.conductance * (x[edge.a] - x[edge.b]);
            out[edge.a] += flux;
            out[edge.b] -= flux;
        }
    };
    let dot = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum()
    };
    let rhs: Vec<f32> = areas
        .iter()
        .zip(source.iter())
        .map(|(&area, &value)| area * value)
        .collect();
    let mut x = source.to_vec();
    let mut ax = vec![0.0f32; n];
    apply(&x, &mut ax);
    let mut residual: Vec<f32> = rhs.iter().zip(ax.iter()).map(|(&b, &a)| b - a).collect();
    let mut direction = residual.clone();
    let mut rr = dot(&residual, &residual);
    let tolerance = 1e-6 * dot(&rhs, &rhs).sqrt().max(1e-20);
    let mut applied = vec![0.0f32; n];
    for _ in 0..256 {
        if rr.sqrt() <= tolerance {
            break;
        }
        apply(&direction, &mut applied);
        let denom = dot(&direction, &applied);
        if denom <= 1e-30 {
            break;
        }
        let alpha = rr / denom;
        for i in 0..n {
            x[i] += (alpha * direction[i] as f64) as f32;
            residual[i] -= (alpha * applied[i] as f64) as f32;
        }
        let next_rr = dot(&residual, &residual);
        let beta = next_rr / rr.max(1e-30);
        for i in 0..n {
            direction[i] = residual[i] + (beta * direction[i] as f64) as f32;
        }
        rr = next_rr;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge() -> SheetEdge {
        SheetEdge {
            a: 0,
            b: 1,
            conductance: 1.0,
            face_length: 1.0,
            normal_a_to_b: Vec3::X,
        }
    }

    #[test]
    fn screened_sheet_transport_conserves_area_weighted_material() {
        let areas = [1.0, 2.0];
        let source = [1.0, 0.0];
        let solved = solve_screened_scalar(&areas, &[edge()], &source, 1.0);
        let before: f32 = source.iter().zip(areas).map(|(h, a)| h * a).sum();
        let after: f32 = solved.iter().zip(areas).map(|(h, a)| h * a).sum();
        assert!((before - after).abs() < 1e-5);
        assert!(solved[0] < source[0] && solved[1] > source[1]);
    }

    #[test]
    fn screened_sheet_preserves_uniform_state() {
        let areas = [1.0, 1.0];
        let source = [0.75, 0.75];
        let solved = solve_screened_scalar(&areas, &[edge()], &source, 3.0);
        assert!((solved[0] - source[0]).abs() < 1e-6);
        assert!((solved[1] - source[1]).abs() < 1e-6);
    }

    /// legacy-yield contract: sub-yield cells are BIT-untouched, over-yield
    /// excess spreads conservatively, and the overload comes down toward yield.
    #[test]
    fn yield_relax_spares_subyield_and_conserves() {
        let areas = [1.0f32; 3];
        // chain 0-1-2
        let edges = [
            SheetEdge {
                a: 0,
                b: 1,
                conductance: 1.0,
                face_length: 1.0,
                normal_a_to_b: Vec3::X,
            },
            SheetEdge {
                a: 1,
                b: 2,
                conductance: 1.0,
                face_length: 1.0,
                normal_a_to_b: Vec3::X,
            },
        ];
        // cell 0 far over yield; cells 1-2 comfortably below.
        let field = [2.0f32, 0.3, 0.1];
        let yield_value = [0.5f32; 3];
        let relaxed = yield_relax_on_edges(&areas, &edges, &field, &yield_value, 2.0, 4);
        let before: f32 = field.iter().zip(areas).map(|(h, a)| h * a).sum();
        let after: f32 = relaxed.iter().zip(areas).map(|(h, a)| h * a).sum();
        assert!(
            (before - after).abs() < 1e-4,
            "volume changed: {before} -> {after}"
        );
        assert!(
            relaxed[0] < field[0],
            "overload not reduced: {}",
            relaxed[0]
        );
        // neighbours receive the shed material (the emergent foothill apron)
        assert!(relaxed[1] > field[1] && relaxed[2] > field[2]);

        // an all-subyield field is returned bit-identical
        let quiet = [0.4f32, 0.3, 0.1];
        let untouched = yield_relax_on_edges(&areas, &edges, &quiet, &yield_value, 2.0, 4);
        assert_eq!(untouched, quiet);

        // width-aware: a lower per-cell threshold on cell 0 sheds more from it
        let tight = [0.2f32, 0.5, 0.5];
        let relaxed_tight = yield_relax_on_edges(&areas, &edges, &field, &tight, 2.0, 4);
        assert!(
            relaxed_tight[0] < relaxed[0],
            "tighter local yield must shed more: {} !< {}",
            relaxed_tight[0],
            relaxed[0]
        );
    }
}
