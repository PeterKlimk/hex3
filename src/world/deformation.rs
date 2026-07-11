//! Coarse thin-sheet crustal deformation.
//!
//! T0 solves a screened horizontal deformation velocity relative to each rigid
//! plate, then updates crust thickness from conservative face fluxes. This is
//! deliberately a velocity/continuity system—not a height kernel. Later rungs
//! can replace the linear viscosity with viscoplastic rheology without changing
//! the state or conservation law.

use glam::Vec3;

use super::boundary::{BoundaryKind, PlateBoundaryEdge, SubductionPolarity};
use super::constants::*;
use super::crust::{Crust, CrustType};
use super::history::{CarrierMesh, CarrierSnapshot};
use super::EulerPole;
use super::{Dynamics, LifecycleAudit, Plates, TectonicHistory, Tessellation};
use glam::Quat;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub(crate) struct ThinSheetFields {
    /// Signed thickness change. Area-weighted integral equals retained magma;
    /// collision shortening itself only redistributes existing crust.
    pub thickness_delta: Vec<f32>,
    /// Accumulated absolute normal strain over the represented episode.
    pub strain: Vec<f32>,
    /// Tangent axis of strongest compression; zero where unresolved.
    pub compression_axis: Vec<Vec3>,
    /// Positive material supplied by retained arc magma (thickness × steradian).
    pub material_added: f64,
    /// Orogenic material transferred into the unresolved sediment reservoir.
    pub material_removed: f64,
    /// Final thickness minus initial, addition, and removal ledgers.
    pub material_residual: f64,
    /// Present physical crust-thickness tendency in thickness units/Myr.
    pub present_uplift_rate: Vec<f32>,
    /// Wall time of the moving carrier deformation solve (projection included).
    pub evolution_seconds: f32,
    /// Fraction of historical receiver-parcel forcing events not represented
    /// by the present-day receiver support.
    pub moving_forcing_fraction: f32,
    pub operator_audit: Option<CarrierOperatorAudit>,
    pub final_continental: Option<Vec<bool>>,
    pub final_ocean_age_myr: Vec<f32>,
    pub final_weakness: Vec<f32>,
    pub lifecycle_audit: Option<LifecycleAudit>,
}

/// Resolution-isolation ladder for the experimental moving-carrier operator.
/// All volume terms are thickness × steradian; maxima are thickness units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CarrierOperatorAudit {
    pub mean_boundary_length_km: f32,
    pub mean_convergent_swept_area_km2_per_myr: f32,
    pub mean_boundary_support_pct: f32,
    pub mean_target_l1: f64,
    pub mean_arc_addition_rate: f64,
    pub one_step_positive: f64,
    pub one_step_negative: f64,
    pub one_step_max: f32,
    pub one_step_transport_max: f32,
    pub one_step_magma_max: f32,
    pub frozen_positive: f64,
    pub frozen_negative: f64,
    pub frozen_max: f32,
    pub moving_positive: f64,
    pub moving_negative: f64,
    pub moving_max: f32,
    pub projected_positive: f64,
    pub projected_negative: f64,
    pub projected_max: f32,
    pub projection_net_residual: f64,
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
        || !field.iter().zip(yield_value.iter()).any(|(&h, &y)| h > y)
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
    let material_added: f64 = magmatic_volume_rate
        .iter()
        .map(|&rate| rate as f64 * duration as f64)
        .sum();
    let material_residual: f64 = thickness_delta
        .iter()
        .zip(areas.iter())
        .map(|(&delta, &area)| delta as f64 * area as f64)
        .sum::<f64>()
        - material_added;

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
        material_added,
        material_removed: 0.0,
        material_residual,
        present_uplift_rate: vec![0.0; n],
        evolution_seconds: 0.0,
        moving_forcing_fraction: 0.0,
        operator_audit: None,
        final_continental: None,
        final_ocean_age_myr: vec![0.0; n],
        final_weakness: vec![0.0; n],
        lifecycle_audit: None,
    }
}

/// Physical-clock thin-sheet evolution. Boundary closure supplies traction,
/// not crustal volume: continuity redistributes the receiving plate's existing
/// crust, while only retained arc magma is a positive material source. Episode
/// start times determine which contacts are active in each interval.
pub(crate) fn solve_history_thin_sheet(
    tess: &Tessellation,
    plates: &Plates,
    crust: &Crust,
    boundaries: &[PlateBoundaryEdge],
    history: &TectonicHistory,
) -> ThinSheetFields {
    let n = tess.num_cells();
    let areas = tess.cell_areas();
    let reference_thickness: Vec<f32> = (0..n)
        .map(|i| match crust.crust_type(i) {
            CrustType::Continental => CRUST_THICKNESS_CONTINENTAL,
            CrustType::Oceanic => CRUST_THICKNESS_OCEANIC,
        })
        .collect();
    let mut thickness = reference_thickness.clone();

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
            let face_length = tess.shared_edge_length(i, j);
            if center_distance <= 1e-8 || face_length <= 0.0 {
                continue;
            }
            let midpoint = (a + b).normalize_or_zero();
            let chord = b - a;
            let normal = (chord - midpoint * midpoint.dot(chord)).normalize_or_zero();
            if normal != Vec3::ZERO {
                edges.push(SheetEdge {
                    a: i,
                    b: j,
                    conductance: face_length / center_distance,
                    face_length,
                    normal_a_to_b: normal,
                });
            }
        }
    }

    let mut ages = vec![0.0f32];
    ages.extend(
        history
            .episodes
            .iter()
            .filter(|episode| episode.duration_myr > 0.0)
            .map(|episode| episode.duration_myr),
    );
    ages.sort_by(f32::total_cmp);
    ages.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let coupling = (HISTORY_SHEET_STRESS_TRANSMISSION_KM / PLANET_RADIUS_KM).powi(2);
    let gravity_diffusivity =
        CRUST_GRAVITATIONAL_DIFFUSIVITY_KM2_PER_MYR / (PLANET_RADIUS_KM * PLANET_RADIUS_KM);
    let rate_scale = MAX_PLATE_ANGULAR_SPEED_RAD_PER_MYR;
    let mut strain = vec![0.0f32; n];
    let mut strongest_compression = vec![0.0f32; n];
    let mut compression_axis = vec![Vec3::ZERO; n];
    let mut total_substeps = 0usize;
    let mut material_added = 0.0f64;

    // Integrate from the oldest represented contact toward the present. A
    // current episode of duration T is active for geological ages [0, T].
    for interval in (1..ages.len()).rev() {
        let age_hi = ages[interval];
        let age_lo = ages[interval - 1];
        let dt_interval = age_hi - age_lo;
        if dt_interval <= 0.0 {
            continue;
        }
        let age_mid = 0.5 * (age_hi + age_lo);
        let mut target_sum = vec![Vec3::ZERO; n];
        let mut target_weight = vec![0.0f32; n];
        let mut magmatic_volume_rate = vec![0.0f32; n];

        for boundary in boundaries {
            let Some(episode) = history.episode_for_edge(boundary.cell_a, boundary.cell_b) else {
                continue;
            };
            if episode.duration_myr < age_mid {
                continue;
            }
            let closing = boundary.convergence.max(0.0) * rate_scale;
            if closing <= 0.0 {
                continue;
            }
            let mut add_target = |cell: usize, magnitude: f32| {
                if magnitude <= 0.0 {
                    return;
                }
                let center = tess.cell_center(cell);
                let toward = boundary.boundary_point - center * center.dot(boundary.boundary_point);
                if toward.length_squared() <= 1e-12 {
                    return;
                }
                let weight = boundary.edge_length;
                target_sum[cell] += -toward.normalize() * magnitude * weight;
                target_weight[cell] += weight;
            };
            match boundary.subduction {
                None if boundary.type_a == CrustType::Continental
                    && boundary.type_b == CrustType::Continental =>
                {
                    add_target(boundary.cell_a, 0.5 * closing);
                    add_target(boundary.cell_b, 0.5 * closing);
                }
                Some(SubductionPolarity::ASubducts) => {
                    if boundary.type_b == CrustType::Continental {
                        add_target(boundary.cell_b, SUBDUCTION_COMPRESSION_COUPLING * closing);
                    }
                    magmatic_volume_rate[boundary.cell_b] += closing
                        * boundary.edge_length
                        * SUBDUCTION_MAGMATIC_ACCRETION
                        * CRUST_THICKNESS_OCEANIC;
                }
                Some(SubductionPolarity::BSubducts) => {
                    if boundary.type_a == CrustType::Continental {
                        add_target(boundary.cell_a, SUBDUCTION_COMPRESSION_COUPLING * closing);
                    }
                    magmatic_volume_rate[boundary.cell_a] += closing
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
        let velocity = solve_velocity(tess, &areas, &edges, &target, coupling);
        let magma_source: Vec<f32> = magmatic_volume_rate
            .iter()
            .zip(areas.iter())
            .map(|(&rate, &area)| rate / area.max(1e-12))
            .collect();
        let magmatic_rate = solve_screened_scalar(&areas, &edges, &magma_source, coupling);
        material_added += magmatic_volume_rate
            .iter()
            .map(|&rate| rate as f64 * dt_interval as f64)
            .sum::<f64>();

        let mut outgoing_rate = vec![0.0f32; n];
        for edge in &edges {
            let midpoint =
                (tess.cell_center(edge.a) + tess.cell_center(edge.b)).normalize_or_zero();
            let va = velocity[edge.a] - midpoint * midpoint.dot(velocity[edge.a]);
            let vb = velocity[edge.b] - midpoint * midpoint.dot(velocity[edge.b]);
            let normal_speed = (0.5 * (va + vb)).dot(edge.normal_a_to_b);
            let (donor, speed) = if normal_speed > 0.0 {
                (edge.a, normal_speed)
            } else {
                (edge.b, -normal_speed)
            };
            outgoing_rate[donor] += speed * edge.face_length / areas[donor].max(1e-12);

            let center_distance = (tess.cell_center(edge.b) - tess.cell_center(edge.a))
                .length()
                .max(1e-8);
            let normal_strain = (vb - va).dot(edge.normal_a_to_b) / center_distance;
            let strain_add = dt_interval * normal_strain.abs();
            strain[edge.a] += 0.5 * strain_add;
            strain[edge.b] += 0.5 * strain_add;
            if normal_strain < -strongest_compression[edge.a] {
                strongest_compression[edge.a] = -normal_strain;
                compression_axis[edge.a] = edge.normal_a_to_b;
            }
            if normal_strain < -strongest_compression[edge.b] {
                strongest_compression[edge.b] = -normal_strain;
                compression_axis[edge.b] = edge.normal_a_to_b;
            }
        }
        let max_courant = dt_interval * outgoing_rate.iter().copied().fold(0.0f32, f32::max);
        let substeps = (max_courant / 0.20).ceil().max(1.0) as usize;
        total_substeps += substeps;
        let dt = dt_interval / substeps as f32;
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

        let tau = gravity_diffusivity * dt_interval;
        if tau > 0.0 {
            thickness = solve_screened_scalar(&areas, &edges, &thickness, tau);
        }
    }

    let thickness_delta: Vec<f32> = thickness
        .iter()
        .zip(reference_thickness.iter())
        .map(|(&after, &before)| after - before)
        .collect();
    let material_residual: f64 = thickness_delta
        .iter()
        .zip(areas.iter())
        .map(|(&delta, &area)| delta as f64 * area as f64)
        .sum::<f64>()
        - material_added;
    log::debug!(
        "history thin sheet: intervals={}, substeps={}, min_H={:.3}, max_H={:.3}, max_dH={:.3}",
        ages.len().saturating_sub(1),
        total_substeps,
        thickness.iter().copied().fold(f32::INFINITY, f32::min),
        thickness.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        thickness_delta
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
    );
    ThinSheetFields {
        thickness_delta,
        strain,
        compression_axis,
        material_added,
        material_removed: 0.0,
        material_residual,
        present_uplift_rate: vec![0.0; n],
        evolution_seconds: 0.0,
        moving_forcing_fraction: 0.0,
        operator_audit: None,
        final_continental: None,
        final_ocean_age_myr: vec![0.0; n],
        final_weakness: vec![0.0; n],
        lifecycle_audit: None,
    }
}

#[derive(Clone, Copy)]
struct CarrierBoundary {
    a: usize,
    b: usize,
    point: Vec3,
    edge_length: f32,
    convergence: f32,
    kind: BoundaryKind,
    polarity: Option<SubductionPolarity>,
    type_a: CrustType,
    type_b: CrustType,
}

#[derive(Default)]
struct CarrierPairStats {
    length: f32,
    normal: f32,
    shear: f32,
    positive_length: f32,
    negative_length: f32,
    ocean_min_vote: f32,
    ocean_max_vote: f32,
}

struct CarrierStep {
    sheet_edges: Vec<SheetEdge>,
    velocity: Vec<Vec3>,
    magmatic_rate: Vec<f32>,
    magmatic_volume_rate: f64,
    strain_rate: Vec<f32>,
    compression_axis: Vec<Vec3>,
    outgoing_rate: Vec<f32>,
    receiver_parcels: HashSet<u16>,
    boundary_length_km: f32,
    convergent_swept_area_km2_per_myr: f32,
    boundary_support_pct: f32,
    target_l1: f64,
}

/// Bounded dynamic-history rung. Every geological interval is forced on that
/// interval's reconstructed carrier domains; conservative thickness state is
/// stored on material parcels, not carrier locations, so it advects naturally
/// to the present without a separate remap.
pub(crate) fn solve_history_carrier_evolved(
    present_tess: &Tessellation,
    dynamics: &Dynamics,
    history: &TectonicHistory,
) -> ThinSheetFields {
    let started = Instant::now();
    let replay = history
        .carrier_replay
        .as_ref()
        .expect("carrier-evolved requires a carrier replay");
    solve_carrier_replay_evolved(present_tess, dynamics, replay, started)
}

#[derive(Clone, Debug)]
struct LifecycleCell {
    plate: usize,
    crust: CrustType,
    ocean_age_myr: f32,
    continental_volume: f64,
    ocean_volume: f64,
    magma_volume: f64,
    continental_area: f64,
    ocean_area: f64,
    underthrust_volume: f64,
    weakness: f32,
    fabric: Vec3,
    collision_deposits: usize,
}

impl LifecycleCell {
    fn total_volume(&self) -> f64 {
        self.continental_volume + self.ocean_volume + self.magma_volume
    }

    fn scale_material(&mut self, fraction: f64) {
        self.continental_volume *= fraction;
        self.ocean_volume *= fraction;
        self.magma_volume *= fraction;
        self.continental_area *= fraction;
        self.ocean_area *= fraction;
        self.underthrust_volume *= fraction;
    }
}

/// Forward lifecycle automaton. The generated carrier layout is interpreted as
/// the oldest state, then advected toward the present. Unlike the back-rotation
/// rungs, gaps and overlaps update material reservoirs and topology.
pub(crate) fn solve_history_carrier_lifecycle(
    present_tess: &Tessellation,
    dynamics: &Dynamics,
    history: &TectonicHistory,
) -> ThinSheetFields {
    let started = Instant::now();
    let replay = history
        .carrier_replay
        .as_ref()
        .expect("carrier lifecycle requires carrier geometry");
    solve_carrier_lifecycle_replay(
        present_tess,
        dynamics,
        replay,
        history.lookback_myr,
        started,
    )
}

fn solve_carrier_lifecycle_replay(
    present_tess: &Tessellation,
    dynamics: &Dynamics,
    replay: &super::history::CarrierReplay,
    lookback_myr: f32,
    started: Instant,
) -> ThinSheetFields {
    let mesh = &replay.mesh;
    let initial = &replay.snapshots[0];
    let n = mesh.centers.len();
    let mut states: Vec<LifecycleCell> = (0..n)
        .map(|cell| {
            let crust = initial.crust_owner[cell];
            let area = mesh.areas[cell] as f64;
            LifecycleCell {
                plate: initial.plate_owner[cell] as usize,
                crust,
                // The generated oldest state has no prior age provenance.
                // Start its ocean clock at zero and advance it physically.
                ocean_age_myr: 0.0,
                continental_volume: if crust == CrustType::Continental {
                    CRUST_THICKNESS_CONTINENTAL as f64 * area
                } else {
                    0.0
                },
                ocean_volume: if crust == CrustType::Oceanic {
                    CRUST_THICKNESS_OCEANIC as f64 * area
                } else {
                    0.0
                },
                magma_volume: 0.0,
                continental_area: if crust == CrustType::Continental {
                    area
                } else {
                    0.0
                },
                ocean_area: if crust == CrustType::Oceanic {
                    area
                } else {
                    0.0
                },
                underthrust_volume: 0.0,
                weakness: 0.0,
                fabric: Vec3::ZERO,
                collision_deposits: 0,
            }
        })
        .collect();
    let initial_material_volume: f64 = states.iter().map(LifecycleCell::total_volume).sum();
    let initial_continental_volume: f64 = states.iter().map(|state| state.continental_volume).sum();
    let plate_count = dynamics.euler_poles.len();
    let mut parent: Vec<usize> = (0..plate_count).collect();
    let mut poles = dynamics.euler_poles.clone();
    let mut collision_closure_km: HashMap<(usize, usize), f32> = HashMap::new();
    let mut audit = LifecycleAudit {
        initial_material_volume,
        ..LifecycleAudit::default()
    };
    let mut previous_thickness = vec![0.0f32; n];
    let steps = (lookback_myr / replay.step_myr).ceil() as usize;

    for step_index in 0..steps {
        let dt = ((step_index + 1) as f32 * replay.step_myr).min(lookback_myr)
            - (step_index as f32 * replay.step_myr).min(lookback_myr);
        if dt <= 0.0 {
            continue;
        }
        for state in &mut states {
            state.plate = lifecycle_find(&mut parent, state.plate);
        }
        age_lifecycle_ocean(&mut states, dt);
        if step_index + 1 == steps {
            previous_thickness = states
                .iter()
                .enumerate()
                .map(|(cell, state)| state.total_volume() as f32 / mesh.areas[cell].max(1e-12))
                .collect();
        }
        let (candidates, admissions) =
            lifecycle_pullback_admissions(mesh, &states, &poles, &mut parent, dt);
        let continental_closure_rates = continental_pair_closure_rates(mesh, &states, &poles);

        let mut resolved: Vec<Option<LifecycleCell>> = vec![None; n];
        let mut step_collision_speed: HashMap<(usize, usize), f32> = HashMap::new();
        for cell in 0..n {
            if admissions[cell].is_empty() {
                continue;
            }
            resolved[cell] = Some(resolve_lifecycle_overlap(
                cell,
                &admissions[cell],
                &candidates,
                mesh,
                &poles,
                &mut parent,
                &continental_closure_rates,
                &mut step_collision_speed,
                &mut audit,
            ));
        }

        // Collision coupling is event-driven: a pair merges only after its
        // accumulated continental closure spans one carrier cell.
        for (pair, speed) in step_collision_speed {
            let closure = collision_closure_km.entry(pair).or_default();
            *closure += speed * dt;
            if *closure >= replay.mean_spacing_km {
                if merge_lifecycle_domains(pair.0, pair.1, &mut parent, &mut poles, &states) {
                    audit.plate_merges += 1;
                    audit.motion_changes += 1;
                }
            }
        }
        for state in resolved.iter_mut().flatten() {
            state.plate = lifecycle_find(&mut parent, state.plate);
        }

        // Only an ocean/ocean divergent hole creates new lithosphere. Other
        // raster holes are conservative domain expansion resolved below.
        for cell in 0..n {
            if resolved[cell].is_some() {
                continue;
            }
            if let Some(plate) = divergent_ocean_gap_owner(cell, mesh, &resolved, &poles) {
                let area = mesh.areas[cell] as f64;
                let new_ocean = new_lifecycle_ocean(plate, area);
                let volume = new_ocean.ocean_volume;
                resolved[cell] = Some(new_ocean);
                audit.created_ocean_area_sr += area;
                audit.created_ocean_volume += volume;
            }
        }

        states = conservatively_fill_lifecycle_gaps(mesh, resolved);
    }

    for state in &mut states {
        state.plate = lifecycle_find(&mut parent, state.plate);
    }
    let final_thickness: Vec<f32> = states
        .iter()
        .enumerate()
        .map(|(cell, state)| state.total_volume() as f32 / mesh.areas[cell].max(1e-12))
        .collect();
    let final_continental_carrier: Vec<bool> = states
        .iter()
        .map(|state| state.crust == CrustType::Continental)
        .collect();
    let reference: Vec<f32> = final_continental_carrier
        .iter()
        .map(|&continental| {
            if continental {
                CRUST_THICKNESS_CONTINENTAL
            } else {
                CRUST_THICKNESS_OCEANIC
            }
        })
        .collect();
    let delta_carrier: Vec<f32> = final_thickness
        .iter()
        .zip(reference.iter())
        .map(|(&after, &before)| after - before)
        .collect();
    let uplift_carrier: Vec<f32> = final_thickness
        .iter()
        .zip(previous_thickness.iter())
        .map(|(&after, &before)| (after - before) / replay.step_myr.max(1e-6))
        .collect();
    let strain_carrier: Vec<f32> = states.iter().map(|state| state.weakness).collect();
    let fabric_carrier: Vec<Vec3> = states.iter().map(|state| state.fabric).collect();
    let age_carrier: Vec<f32> = states
        .iter()
        .map(|state| {
            if state.crust == CrustType::Oceanic {
                state.ocean_age_myr
            } else {
                0.0
            }
        })
        .collect();

    audit.final_material_volume = states.iter().map(LifecycleCell::total_volume).sum();
    audit.material_residual =
        audit.final_material_volume - audit.initial_material_volume - audit.created_ocean_volume
            + audit.consumed_ocean_volume
            - audit.magmatic_added_volume;
    let final_continental_volume: f64 = states.iter().map(|state| state.continental_volume).sum();
    audit.continental_material_residual = final_continental_volume - initial_continental_volume;
    audit.active_sutures = count_lifecycle_sutures(mesh, &states);
    audit.final_plate_count = states
        .iter()
        .map(|state| state.plate)
        .collect::<HashSet<_>>()
        .len();
    audit.final_unresolved_overlaps = 0;
    audit.final_zero_age_ocean_cells = states
        .iter()
        .filter(|state| state.crust == CrustType::Oceanic && state.ocean_age_myr == 0.0)
        .count();
    let mut sorted_thickness = final_thickness.clone();
    sorted_thickness.sort_by(f32::total_cmp);
    let quantile =
        |p: f32| sorted_thickness[(((sorted_thickness.len() - 1) as f32) * p).round() as usize];
    audit.thickness_p50 = quantile(0.50);
    audit.thickness_p90 = quantile(0.90);
    audit.thickness_p99 = quantile(0.99);
    audit.carrier_max_thickness = final_thickness
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    audit.carrier_max_delta = delta_carrier
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let total_area: f64 = mesh.areas.iter().map(|&area| area as f64).sum();
    audit.positive_delta_area_fraction = delta_carrier
        .iter()
        .zip(mesh.areas.iter())
        .filter(|&(&delta, _)| delta > 0.0)
        .map(|(_, &area)| area as f64)
        .sum::<f64>() as f32
        / total_area.max(1e-30) as f32;
    for cell in 0..n {
        let area = mesh.areas[cell].max(1e-12);
        let underthrust = states[cell].underthrust_volume as f32 / area;
        let magma = states[cell].magma_volume as f32 / area;
        let remap = delta_carrier[cell] - underthrust - magma;
        audit.underthrust_positive_volume += states[cell].underthrust_volume;
        audit.magma_positive_volume += states[cell].magma_volume;
        audit.remap_positive_volume += remap.max(0.0) as f64 * area as f64;
        audit.max_underthrust_thickness = audit.max_underthrust_thickness.max(underthrust);
        audit.max_magma_thickness = audit.max_magma_thickness.max(magma);
        audit.max_remap_thickness = audit.max_remap_thickness.max(remap);
        audit.max_collision_deposits = audit
            .max_collision_deposits
            .max(states[cell].collision_deposits);
    }
    let thickness_delta = project_carrier_scalar(present_tess, mesh, &delta_carrier);
    audit.projected_max_delta = thickness_delta
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let strain = project_carrier_scalar(present_tess, mesh, &strain_carrier);
    let compression_axis = project_carrier_vec3(present_tess, mesh, &fabric_carrier);
    let present_uplift_rate = project_carrier_scalar(present_tess, mesh, &uplift_carrier);
    let final_continental = Some(project_carrier_bool(
        present_tess,
        mesh,
        &final_continental_carrier,
    ));
    let final_ocean_age_myr = project_carrier_scalar(present_tess, mesh, &age_carrier);
    let final_weakness = strain.clone();
    audit.runtime_seconds = started.elapsed().as_secs_f32();

    ThinSheetFields {
        thickness_delta,
        strain,
        compression_axis,
        material_added: audit.magmatic_added_volume + audit.created_ocean_volume,
        material_removed: audit.consumed_ocean_volume,
        material_residual: audit.material_residual,
        present_uplift_rate,
        evolution_seconds: audit.runtime_seconds,
        moving_forcing_fraction: 0.0,
        operator_audit: None,
        final_continental,
        final_ocean_age_myr,
        final_weakness,
        lifecycle_audit: Some(audit),
    }
}

fn solve_carrier_replay_evolved(
    present_tess: &Tessellation,
    _dynamics: &Dynamics,
    replay: &super::history::CarrierReplay,
    started: Instant,
) -> ThinSheetFields {
    let mesh = &replay.mesh;
    let n = mesh.centers.len();
    let present = &replay.snapshots[0];
    debug_assert_eq!(n, present.surface_parcel.len());

    let parcel_crust: Vec<_> = (0..n).map(|parcel| present.crust_owner[parcel]).collect();
    let reference_thickness: Vec<_> = parcel_crust
        .iter()
        .map(|crust| match crust {
            CrustType::Continental => CRUST_THICKNESS_CONTINENTAL,
            CrustType::Oceanic => CRUST_THICKNESS_OCEANIC,
        })
        .collect();
    let mut parcel_volume: Vec<f32> = reference_thickness
        .iter()
        .zip(mesh.areas.iter())
        .map(|(&thickness, &area)| thickness * area)
        .collect();
    let initial_volume: f64 = parcel_volume.iter().map(|&volume| volume as f64).sum();
    let mut parcel_strain = vec![0.0f32; n];
    let mut strongest_compression = vec![0.0f32; n];
    let mut parcel_axis = vec![Vec3::ZERO; n];
    let current_receivers = carrier_receiver_parcels(mesh, present);
    let mut forcing_events = 0usize;
    let mut moving_forcing_events = 0usize;
    let mut material_added = 0.0f64;
    let mut material_removed = 0.0f64;
    let mut audit_boundary_length = 0.0f64;
    let mut audit_swept_area = 0.0f64;
    let mut audit_boundary_support = 0.0f64;
    let mut audit_target_l1 = 0.0f64;
    let mut audit_arc_rate = 0.0f64;
    let mut audit_steps = 0usize;

    // Snapshots are stored present -> past. Integrate oldest -> present so each
    // parcel carries inherited state through the changing surface ownership.
    for index in (1..replay.snapshots.len()).rev() {
        let snapshot = &replay.snapshots[index];
        let younger = &replay.snapshots[index - 1];
        let dt = snapshot.lookback_myr - younger.lookback_myr;
        if dt <= 0.0 {
            continue;
        }
        let (mut thickness, owned_area) = distribute_parcel_volume(mesh, snapshot, &parcel_volume);
        let step = carrier_step(mesh, snapshot);
        audit_boundary_length += step.boundary_length_km as f64;
        audit_swept_area += step.convergent_swept_area_km2_per_myr as f64;
        audit_boundary_support += step.boundary_support_pct as f64;
        audit_target_l1 += step.target_l1;
        audit_arc_rate += step.magmatic_volume_rate;
        audit_steps += 1;
        forcing_events += step.receiver_parcels.len();
        moving_forcing_events += step.receiver_parcels.difference(&current_receivers).count();
        material_added += step.magmatic_volume_rate * dt as f64;

        let max_courant = dt * step.outgoing_rate.iter().copied().fold(0.0f32, f32::max);
        let substeps = (max_courant / 0.20).ceil().max(1.0) as usize;
        let sub_dt = dt / substeps as f32;
        let mut volume_change = vec![0.0f32; n];
        for _ in 0..substeps {
            volume_change.fill(0.0);
            for edge in &step.sheet_edges {
                let midpoint = (mesh.centers[edge.a] + mesh.centers[edge.b]).normalize_or_zero();
                let va = step.velocity[edge.a] - midpoint * midpoint.dot(step.velocity[edge.a]);
                let vb = step.velocity[edge.b] - midpoint * midpoint.dot(step.velocity[edge.b]);
                let speed = (0.5 * (va + vb)).dot(edge.normal_a_to_b);
                let donor_h = if speed >= 0.0 {
                    thickness[edge.a]
                } else {
                    thickness[edge.b]
                };
                let flux = donor_h * speed * edge.face_length;
                volume_change[edge.a] -= sub_dt * flux;
                volume_change[edge.b] += sub_dt * flux;
            }
            for cell in 0..n {
                thickness[cell] += volume_change[cell] / mesh.areas[cell].max(1e-12);
                thickness[cell] += sub_dt * step.magmatic_rate[cell].max(0.0);
            }
        }

        let gravity_tau = CRUST_GRAVITATIONAL_DIFFUSIVITY_KM2_PER_MYR
            / (PLANET_RADIUS_KM * PLANET_RADIUS_KM)
            * dt;
        if gravity_tau > 0.0 {
            thickness =
                solve_screened_scalar(&mesh.areas, &step.sheet_edges, &thickness, gravity_tau);
        }
        if replay.denudation_rate_km_per_myr > 0.0 {
            let surface_reference: Vec<_> = snapshot
                .surface_parcel
                .iter()
                .map(|&parcel| reference_thickness[parcel as usize])
                .collect();
            material_removed += denude_excess_thickness(
                &mut thickness,
                &surface_reference,
                &mesh.areas,
                replay.denudation_rate_km_per_myr,
                dt,
            );
        }
        gather_visible_parcel_volume(mesh, snapshot, &owned_area, &thickness, &mut parcel_volume);

        for cell in 0..n {
            let parcel = snapshot.surface_parcel[cell] as usize;
            let weight = mesh.areas[cell] / owned_area[parcel].max(1e-12);
            parcel_strain[parcel] += dt * step.strain_rate[cell] * weight;
            if step.strain_rate[cell] > strongest_compression[parcel] {
                strongest_compression[parcel] = step.strain_rate[cell];
                let plate = snapshot.plate_owner[cell] as usize;
                // Historical tangent fabric is carried forward with its plate.
                parcel_axis[parcel] =
                    snapshot.plate_past_to_present[plate] * step.compression_axis[cell];
            }
        }
    }

    let final_thickness: Vec<_> = parcel_volume
        .iter()
        .zip(mesh.areas.iter())
        .map(|(&volume, &area)| volume / area.max(1e-12))
        .collect();
    let parcel_delta: Vec<_> = final_thickness
        .iter()
        .zip(reference_thickness.iter())
        .map(|(&after, &before)| after - before)
        .collect();
    let final_volume: f64 = parcel_volume.iter().map(|&volume| volume as f64).sum();
    let material_residual = final_volume - initial_volume - material_added + material_removed;

    // A final non-integrated solve exports the current physical thickness
    // tendency. This is provenance for future T3 erosion; it is not fed back
    // into the legacy dimensionless erosion clock.
    let (present_thickness, present_owned_area) =
        distribute_parcel_volume(mesh, present, &parcel_volume);
    let present_step = carrier_step(mesh, present);
    let cell_volume_rate = carrier_cell_volume_rate(mesh, &present_thickness, &present_step);
    let mut parcel_uplift_rate = vec![0.0f32; n];
    for cell in 0..n {
        let parcel = present.surface_parcel[cell] as usize;
        parcel_uplift_rate[parcel] += cell_volume_rate[cell] / mesh.areas[parcel].max(1e-12);
    }
    debug_assert!(present_owned_area.iter().all(|&area| area > 0.0));

    let thickness_delta = project_carrier_scalar(present_tess, mesh, &parcel_delta);
    let strain = project_carrier_scalar(present_tess, mesh, &parcel_strain);
    let compression_axis = project_carrier_vec3(present_tess, mesh, &parcel_axis);
    let present_uplift_rate = project_carrier_scalar(present_tess, mesh, &parcel_uplift_rate);
    let moving_forcing_fraction = moving_forcing_events as f32 / forcing_events.max(1) as f32;
    let operator_audit = if replay.operator_audit {
        let reference_cells: Vec<_> = present
            .crust_owner
            .iter()
            .map(|crust| match crust {
                CrustType::Continental => CRUST_THICKNESS_CONTINENTAL,
                CrustType::Oceanic => CRUST_THICKNESS_OCEANIC,
            })
            .collect();
        let one_step = evolve_frozen_carrier(
            mesh,
            &present_step,
            &reference_cells,
            replay.step_myr,
            1,
            true,
            true,
        );
        let one_step_transport = evolve_frozen_carrier(
            mesh,
            &present_step,
            &reference_cells,
            replay.step_myr,
            1,
            true,
            false,
        );
        let one_step_magma = evolve_frozen_carrier(
            mesh,
            &present_step,
            &reference_cells,
            replay.step_myr,
            1,
            false,
            true,
        );
        let frozen_steps = (history_duration_myr(replay) / replay.step_myr)
            .round()
            .max(1.0) as usize;
        let frozen = evolve_frozen_carrier(
            mesh,
            &present_step,
            &reference_cells,
            replay.step_myr,
            frozen_steps,
            true,
            true,
        );
        let (one_positive, one_negative, one_max, _) =
            thickness_change_stats(&one_step, &reference_cells, &mesh.areas);
        let (frozen_positive, frozen_negative, frozen_max, _) =
            thickness_change_stats(&frozen, &reference_cells, &mesh.areas);
        let (_, _, one_step_transport_max, _) =
            thickness_change_stats(&one_step_transport, &reference_cells, &mesh.areas);
        let (_, _, one_step_magma_max, _) =
            thickness_change_stats(&one_step_magma, &reference_cells, &mesh.areas);
        let (moving_positive, moving_negative, moving_max, moving_net) =
            field_stats(&parcel_delta, &mesh.areas);
        let (projected_positive, projected_negative, projected_max, projected_net) =
            field_stats(&thickness_delta, &present_tess.cell_areas());
        let inv_audit_steps = 1.0 / audit_steps.max(1) as f64;
        Some(CarrierOperatorAudit {
            mean_boundary_length_km: (audit_boundary_length * inv_audit_steps) as f32,
            mean_convergent_swept_area_km2_per_myr: (audit_swept_area * inv_audit_steps) as f32,
            mean_boundary_support_pct: (audit_boundary_support * inv_audit_steps) as f32,
            mean_target_l1: audit_target_l1 * inv_audit_steps,
            mean_arc_addition_rate: audit_arc_rate * inv_audit_steps,
            one_step_positive: one_positive,
            one_step_negative: one_negative,
            one_step_max: one_max,
            one_step_transport_max,
            one_step_magma_max,
            frozen_positive,
            frozen_negative,
            frozen_max,
            moving_positive,
            moving_negative,
            moving_max,
            projected_positive,
            projected_negative,
            projected_max,
            projection_net_residual: projected_net - moving_net,
        })
    } else {
        None
    };

    ThinSheetFields {
        thickness_delta,
        strain,
        compression_axis,
        material_added,
        material_removed,
        material_residual,
        present_uplift_rate,
        evolution_seconds: started.elapsed().as_secs_f32(),
        moving_forcing_fraction,
        operator_audit,
        final_continental: None,
        final_ocean_age_myr: vec![0.0; present_tess.num_cells()],
        final_weakness: vec![0.0; present_tess.num_cells()],
        lifecycle_audit: None,
    }
}

/// Remove only tectonic thickness above the local undeformed column. The rate
/// is a surface-lowering capacity: Airy isostasy converts it to crust-thickness
/// loss. Removed material remains explicit in the sediment-export ledger rather
/// than being silently destroyed or painted into basins that this coarse rung
/// cannot resolve.
fn denude_excess_thickness(
    thickness: &mut [f32],
    reference: &[f32],
    areas: &[f32],
    surface_rate_km_per_myr: f32,
    dt_myr: f32,
) -> f64 {
    debug_assert_eq!(thickness.len(), reference.len());
    debug_assert_eq!(thickness.len(), areas.len());
    if surface_rate_km_per_myr <= 0.0 || dt_myr <= 0.0 {
        return 0.0;
    }
    let isostasy_slope = (CONTINENTAL_BASE - ABYSSAL_DEPTH)
        / (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC);
    let thickness_capacity =
        surface_rate_km_per_myr * dt_myr / (ELEVATION_UNIT_KM * isostasy_slope);
    thickness
        .iter_mut()
        .zip(reference.iter())
        .zip(areas.iter())
        .map(|((value, &base), &area)| {
            let removed = (*value - base).max(0.0).min(thickness_capacity);
            *value -= removed;
            removed as f64 * area as f64
        })
        .sum()
}

fn lifecycle_find(parent: &mut [usize], mut node: usize) -> usize {
    let mut root = node;
    while parent[root] != root {
        root = parent[root];
    }
    while parent[node] != node {
        let next = parent[node];
        parent[node] = root;
        node = next;
    }
    root
}

fn age_lifecycle_ocean(states: &mut [LifecycleCell], dt_myr: f32) {
    for state in states {
        if state.crust == CrustType::Oceanic {
            state.ocean_age_myr += dt_myr;
        }
    }
}

fn new_lifecycle_ocean(plate: usize, area: f64) -> LifecycleCell {
    LifecycleCell {
        plate,
        crust: CrustType::Oceanic,
        ocean_age_myr: 0.0,
        continental_volume: 0.0,
        ocean_volume: CRUST_THICKNESS_OCEANIC as f64 * area,
        magma_volume: 0.0,
        continental_area: 0.0,
        ocean_area: area,
        underthrust_volume: 0.0,
        weakness: 0.0,
        fabric: Vec3::ZERO,
        collision_deposits: 0,
    }
}

fn merge_lifecycle_domains(
    a: usize,
    b: usize,
    parent: &mut [usize],
    poles: &mut [EulerPole],
    states: &[LifecycleCell],
) -> bool {
    let root_a = lifecycle_find(parent, a);
    let root_b = lifecycle_find(parent, b);
    if root_a == root_b {
        return false;
    }
    let (keep, merge) = if root_a < root_b {
        (root_a, root_b)
    } else {
        (root_b, root_a)
    };
    let weight = |root: usize| -> f32 {
        states
            .iter()
            .filter(|state| state.plate == root)
            .map(|state| (state.continental_area + state.ocean_area) as f32)
            .sum::<f32>()
            .max(1e-9)
    };
    let keep_weight = weight(keep);
    let merge_weight = weight(merge);
    let omega = (poles[keep].axis * poles[keep].angular_velocity * keep_weight
        + poles[merge].axis * poles[merge].angular_velocity * merge_weight)
        / (keep_weight + merge_weight);
    let magnitude = omega.length();
    poles[keep] = EulerPole {
        axis: if magnitude > 1e-9 {
            omega / magnitude
        } else {
            poles[keep].axis
        },
        angular_velocity: magnitude.min(MAX_ANGULAR_VELOCITY),
    };
    parent[merge] = keep;
    true
}

/// Topology-aware semi-Lagrangian pullback. Each destination asks every active
/// motion domain where it came from; a plate is admitted only when that source
/// cell belonged to the plate. Per-component plate totals are normalized after
/// sampling, so pullback cannot duplicate or destroy material and a uniform
/// rigidly rotating plate remains uniform up to carrier-area roundoff.
fn lifecycle_pullback_admissions(
    mesh: &CarrierMesh,
    states: &[LifecycleCell],
    poles: &[EulerPole],
    parent: &mut [usize],
    dt_myr: f32,
) -> (Vec<LifecycleCell>, Vec<Vec<usize>>) {
    #[derive(Default, Clone, Copy)]
    struct Totals {
        continental_volume: f64,
        ocean_volume: f64,
        magma_volume: f64,
        continental_area: f64,
        ocean_area: f64,
        underthrust_volume: f64,
    }
    let mut active: Vec<_> = states.iter().map(|state| state.plate).collect();
    active.sort_unstable();
    active.dedup();
    let mut source_totals: HashMap<usize, Totals> = HashMap::new();
    for state in states {
        let total = source_totals.entry(state.plate).or_default();
        total.continental_volume += state.continental_volume;
        total.ocean_volume += state.ocean_volume;
        total.magma_volume += state.magma_volume;
        total.continental_area += state.continental_area;
        total.ocean_area += state.ocean_area;
        total.underthrust_volume += state.underthrust_volume;
    }

    let mut candidates = Vec::new();
    let mut candidate_sources = Vec::new();
    let mut admissions = vec![Vec::new(); mesh.centers.len()];
    for destination in 0..mesh.centers.len() {
        for &plate in &active {
            let plate = lifecycle_find(parent, plate);
            let source_point = Quat::from_axis_angle(
                poles[plate].axis,
                -poles[plate].angular_velocity_rad_per_myr() * dt_myr,
            ) * mesh.centers[destination];
            let source = nearest_carrier_cell(mesh, source_point, destination);
            if states[source].plate != plate {
                continue;
            }
            let mut candidate = states[source].clone();
            candidate.plate = plate;
            candidate.fabric = (Quat::from_axis_angle(
                poles[plate].axis,
                poles[plate].angular_velocity_rad_per_myr() * dt_myr,
            ) * candidate.fabric)
                .normalize_or_zero();
            candidate.scale_material(
                mesh.areas[destination] as f64 / mesh.areas[source].max(1e-12) as f64,
            );
            let id = candidates.len();
            candidates.push(candidate);
            candidate_sources.push(source);
            admissions[destination].push(id);
        }
    }

    // A sub-cell motion domain can otherwise receive no pullback samples and
    // vanish numerically. Give each unsampled active domain one deterministic
    // forward-mapped support cell; it may then disappear only through the
    // explicit overlap consumption/merge rules below.
    let sampled_plates: HashSet<_> = candidates.iter().map(|state| state.plate).collect();
    for &plate in &active {
        if sampled_plates.contains(&plate) {
            continue;
        }
        let source = states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.plate == plate)
            .max_by(|(a, state_a), (b, state_b)| {
                (state_a.continental_area + state_a.ocean_area)
                    .total_cmp(&(state_b.continental_area + state_b.ocean_area))
                    .then_with(|| b.cmp(a))
            })
            .map(|(cell, _)| cell)
            .expect("active plate has material");
        let destination_point = Quat::from_axis_angle(
            poles[plate].axis,
            poles[plate].angular_velocity_rad_per_myr() * dt_myr,
        ) * mesh.centers[source];
        let destination = nearest_carrier_cell(mesh, destination_point, source);
        let mut candidate = states[source].clone();
        candidate.fabric = (Quat::from_axis_angle(
            poles[plate].axis,
            poles[plate].angular_velocity_rad_per_myr() * dt_myr,
        ) * candidate.fabric)
            .normalize_or_zero();
        candidate
            .scale_material(mesh.areas[destination] as f64 / mesh.areas[source].max(1e-12) as f64);
        let id = candidates.len();
        candidates.push(candidate);
        candidate_sources.push(source);
        admissions[destination].push(id);
    }

    // Persistent weakness is an intensive damage field, not material volume.
    // Track each connected suture component independently so a narrow component
    // cannot disappear merely because nearest pullback missed its one cell.
    let mut seen_weak = vec![false; states.len()];
    for start in 0..states.len() {
        if seen_weak[start] || states[start].weakness <= 0.0 {
            continue;
        }
        seen_weak[start] = true;
        let plate = states[start].plate;
        let mut queue = std::collections::VecDeque::from([start]);
        let mut component = Vec::new();
        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            for &next in &mesh.neighbors[cell] {
                let next = next as usize;
                if !seen_weak[next] && states[next].plate == plate && states[next].weakness > 0.0 {
                    seen_weak[next] = true;
                    queue.push_back(next);
                }
            }
        }
        let representative = component
            .iter()
            .copied()
            .max_by(|&a, &b| {
                states[a]
                    .weakness
                    .total_cmp(&states[b].weakness)
                    .then_with(|| {
                        states[a]
                            .collision_deposits
                            .cmp(&states[b].collision_deposits)
                    })
                    .then_with(|| b.cmp(&a))
            })
            .unwrap();
        let component_set: HashSet<_> = component.iter().copied().collect();
        let candidate = candidate_sources
            .iter()
            .enumerate()
            .filter(|(_, source)| component_set.contains(source))
            .map(|(id, _)| id)
            .next()
            .or_else(|| {
                let forward = Quat::from_axis_angle(
                    poles[plate].axis,
                    poles[plate].angular_velocity_rad_per_myr() * dt_myr,
                ) * mesh.centers[representative];
                let destination = nearest_carrier_cell(mesh, forward, representative);
                admissions[destination]
                    .iter()
                    .copied()
                    .find(|&id| candidates[id].plate == plate)
            })
            .or_else(|| candidates.iter().position(|state| state.plate == plate))
            .expect("weak component plate retains pullback support");
        let max_weakness = component
            .iter()
            .map(|&cell| states[cell].weakness)
            .fold(0.0f32, f32::max);
        let max_deposits = component
            .iter()
            .map(|&cell| states[cell].collision_deposits)
            .max()
            .unwrap_or(0);
        candidates[candidate].weakness = candidates[candidate].weakness.max(max_weakness);
        candidates[candidate].collision_deposits =
            candidates[candidate].collision_deposits.max(max_deposits);
        candidates[candidate].fabric = (Quat::from_axis_angle(
            poles[plate].axis,
            poles[plate].angular_velocity_rad_per_myr() * dt_myr,
        ) * states[representative].fabric)
            .normalize_or_zero();
    }

    let mut sampled_totals: HashMap<usize, Totals> = HashMap::new();
    for state in &candidates {
        let total = sampled_totals.entry(state.plate).or_default();
        total.continental_volume += state.continental_volume;
        total.ocean_volume += state.ocean_volume;
        total.magma_volume += state.magma_volume;
        total.continental_area += state.continental_area;
        total.ocean_area += state.ocean_area;
        total.underthrust_volume += state.underthrust_volume;
    }
    // Nearest pullback can miss a sub-cell material component (most commonly a
    // narrow magma or underthrust streak) even while sampling its host plate.
    // Preserve that ledger on the plate's first deterministic support cell;
    // this is a conservative fallback, not a new surface source.
    for (&plate, source) in &source_totals {
        let sampled = sampled_totals.entry(plate).or_default();
        let first = candidates
            .iter()
            .position(|state| state.plate == plate)
            .expect("active plate has pullback support");
        if source.continental_volume > 0.0 && sampled.continental_volume == 0.0 {
            candidates[first].continental_volume = source.continental_volume;
            sampled.continental_volume = source.continental_volume;
        }
        if source.ocean_volume > 0.0 && sampled.ocean_volume == 0.0 {
            candidates[first].ocean_volume = source.ocean_volume;
            sampled.ocean_volume = source.ocean_volume;
        }
        if source.magma_volume > 0.0 && sampled.magma_volume == 0.0 {
            candidates[first].magma_volume = source.magma_volume;
            sampled.magma_volume = source.magma_volume;
        }
        if source.continental_area > 0.0 && sampled.continental_area == 0.0 {
            candidates[first].continental_area = source.continental_area;
            sampled.continental_area = source.continental_area;
        }
        if source.ocean_area > 0.0 && sampled.ocean_area == 0.0 {
            candidates[first].ocean_area = source.ocean_area;
            sampled.ocean_area = source.ocean_area;
        }
        if source.underthrust_volume > 0.0 && sampled.underthrust_volume == 0.0 {
            candidates[first].underthrust_volume = source.underthrust_volume;
            sampled.underthrust_volume = source.underthrust_volume;
        }
    }
    let ratio = |target: f64, sampled: f64| {
        if sampled.abs() > 1e-30 {
            target / sampled
        } else {
            0.0
        }
    };
    for state in &mut candidates {
        let source = source_totals[&state.plate];
        let sampled = sampled_totals[&state.plate];
        state.continental_volume *= ratio(source.continental_volume, sampled.continental_volume);
        state.ocean_volume *= ratio(source.ocean_volume, sampled.ocean_volume);
        state.magma_volume *= ratio(source.magma_volume, sampled.magma_volume);
        state.continental_area *= ratio(source.continental_area, sampled.continental_area);
        state.ocean_area *= ratio(source.ocean_area, sampled.ocean_area);
        state.underthrust_volume *= ratio(source.underthrust_volume, sampled.underthrust_volume);
    }
    (candidates, admissions)
}

/// Edge-length-weighted positive normal convergence on actual continental
/// contacts before remap. Transform path length and divergent motion contribute
/// exactly zero to the merge clock.
fn continental_pair_closure_rates(
    mesh: &CarrierMesh,
    states: &[LifecycleCell],
    poles: &[EulerPole],
) -> HashMap<(usize, usize), f32> {
    let mut sums: HashMap<(usize, usize), (f32, f32)> = HashMap::new();
    for edge in &mesh.edges {
        let a = &states[edge.a];
        let b = &states[edge.b];
        if a.crust != CrustType::Continental
            || b.crust != CrustType::Continental
            || a.plate == b.plate
        {
            continue;
        }
        let point = (mesh.centers[edge.a] + mesh.centers[edge.b]).normalize_or_zero();
        let relative = poles[a.plate].velocity_at(point) - poles[b.plate].velocity_at(point);
        let normal = relative.dot(edge.normal_a_to_b);
        let closure = if normal > TRANSFORM_NORMAL_THRESHOLD {
            normal * MAX_PLATE_SPEED_KM_PER_MYR
        } else {
            0.0
        };
        let entry = sums
            .entry(canonical_plate_pair(a.plate, b.plate))
            .or_default();
        entry.0 += closure * edge.face_length;
        entry.1 += edge.face_length;
    }
    sums.into_iter()
        .filter_map(|(pair, (closure, length))| {
            (length > 0.0 && closure > 0.0).then_some((pair, closure / length))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn resolve_lifecycle_overlap(
    cell: usize,
    sources: &[usize],
    states: &[LifecycleCell],
    mesh: &CarrierMesh,
    poles: &[EulerPole],
    parent: &mut [usize],
    continental_closure_rates: &HashMap<(usize, usize), f32>,
    step_collision_speed: &mut HashMap<(usize, usize), f32>,
    audit: &mut LifecycleAudit,
) -> LifecycleCell {
    let mut continental: Vec<_> = sources
        .iter()
        .copied()
        .filter(|&source| states[source].crust == CrustType::Continental)
        .collect();
    let mut oceanic: Vec<_> = sources
        .iter()
        .copied()
        .filter(|&source| states[source].crust == CrustType::Oceanic)
        .collect();
    continental.sort_unstable();
    oceanic.sort_by(|&a, &b| {
        states[a]
            .ocean_age_myr
            .total_cmp(&states[b].ocean_age_myr)
            .then_with(|| a.cmp(&b))
    });

    if !continental.is_empty() {
        let winner_source = continental[0];
        let mut winner = states[winner_source].clone();
        winner.plate = lifecycle_find(parent, winner.plate);
        for &source in continental.iter().skip(1) {
            let loser = &states[source];
            let loser_plate = lifecycle_find(parent, loser.plate);
            if loser_plate != winner.plate {
                let pair = canonical_plate_pair(winner.plate, loser_plate);
                if let Some(&closure_rate) = continental_closure_rates.get(&pair) {
                    if closure_rate > 0.0 {
                        step_collision_speed
                            .entry(pair)
                            .and_modify(|value| *value = value.max(closure_rate))
                            .or_insert(closure_rate);
                    }
                }
                winner.weakness = 1.0;
                winner.underthrust_volume += loser.continental_volume;
                winner.collision_deposits += 1;
                audit.continental_underthrust_volume += loser.continental_volume;
                let relative = poles[winner.plate].velocity_km_per_myr_at(mesh.centers[cell])
                    - poles[loser_plate].velocity_km_per_myr_at(mesh.centers[cell]);
                winner.fabric = relative.normalize_or_zero();
            }
            winner.continental_volume += loser.continental_volume;
            winner.ocean_volume += loser.ocean_volume;
            winner.magma_volume += loser.magma_volume;
            winner.continental_area += loser.continental_area;
            winner.ocean_area += loser.ocean_area;
            winner.underthrust_volume += loser.underthrust_volume;
            winner.collision_deposits += loser.collision_deposits;
            winner.weakness = winner.weakness.max(loser.weakness);
        }
        for source in oceanic {
            let loser = &states[source];
            let loser_plate = lifecycle_find(parent, loser.plate);
            if loser_plate == winner.plate {
                // Same-motion raster alias: retain material explicitly rather
                // than pretending it subducted.
                winner.ocean_volume += loser.ocean_volume;
                winner.ocean_area += loser.ocean_area;
                winner.magma_volume += loser.magma_volume;
                winner.continental_volume += loser.continental_volume;
                winner.continental_area += loser.continental_area;
                winner.underthrust_volume += loser.underthrust_volume;
                continue;
            }
            winner.continental_volume += loser.continental_volume;
            winner.continental_area += loser.continental_area;
            winner.underthrust_volume += loser.underthrust_volume;
            audit.consumed_ocean_area_sr += loser.ocean_area;
            let consumed = loser.ocean_volume + loser.magma_volume;
            audit.consumed_ocean_volume += consumed;
            let magma = consumed * SUBDUCTION_MAGMATIC_ACCRETION as f64;
            winner.magma_volume += magma;
            audit.magmatic_added_volume += magma;
        }
        winner.crust = CrustType::Continental;
        winner.ocean_age_myr = 0.0;
        return winner;
    }

    let winner_source = oceanic[0];
    let mut winner = states[winner_source].clone();
    winner.plate = lifecycle_find(parent, winner.plate);
    let mut retained_age_moment = winner.ocean_age_myr as f64 * winner.ocean_volume;
    for &source in oceanic.iter().skip(1) {
        let loser = &states[source];
        let loser_plate = lifecycle_find(parent, loser.plate);
        if loser_plate == winner.plate {
            retained_age_moment += loser.ocean_age_myr as f64 * loser.ocean_volume;
            winner.ocean_volume += loser.ocean_volume;
            winner.ocean_area += loser.ocean_area;
            winner.magma_volume += loser.magma_volume;
            winner.continental_volume += loser.continental_volume;
            winner.continental_area += loser.continental_area;
            winner.underthrust_volume += loser.underthrust_volume;
        } else {
            winner.continental_volume += loser.continental_volume;
            winner.continental_area += loser.continental_area;
            winner.underthrust_volume += loser.underthrust_volume;
            audit.consumed_ocean_area_sr += loser.ocean_area;
            let consumed = loser.ocean_volume + loser.magma_volume;
            audit.consumed_ocean_volume += consumed;
            let magma = consumed * SUBDUCTION_MAGMATIC_ACCRETION as f64;
            winner.magma_volume += magma;
            audit.magmatic_added_volume += magma;
        }
    }
    winner.ocean_age_myr = (retained_age_moment / winner.ocean_volume.max(1e-30)) as f32;
    if winner.continental_volume > 0.0 {
        winner.crust = CrustType::Continental;
        winner.weakness = winner.weakness.max(1.0);
    }
    winner
}

fn divergent_ocean_gap_owner(
    cell: usize,
    mesh: &CarrierMesh,
    states: &[Option<LifecycleCell>],
    poles: &[EulerPole],
) -> Option<usize> {
    let occupied: Vec<_> = mesh.neighbors[cell]
        .iter()
        .filter_map(|&neighbor| {
            let neighbor = neighbor as usize;
            states[neighbor].as_ref().map(|state| (neighbor, state))
        })
        .filter(|(_, state)| state.crust == CrustType::Oceanic)
        .collect();
    let mut best: Option<(f32, usize)> = None;
    for i in 0..occupied.len() {
        for j in i + 1..occupied.len() {
            let (a, state_a) = occupied[i];
            let (b, state_b) = occupied[j];
            if state_a.plate == state_b.plate {
                continue;
            }
            let point = mesh.centers[cell];
            let chord = mesh.centers[b] - mesh.centers[a];
            let normal = (chord - point * point.dot(chord)).normalize_or_zero();
            let relative =
                poles[state_a.plate].velocity_at(point) - poles[state_b.plate].velocity_at(point);
            let convergence = relative.dot(normal);
            if convergence < -TRANSFORM_NORMAL_THRESHOLD {
                let owner = state_a.plate.min(state_b.plate);
                if best.map_or(true, |(value, _)| convergence < value) {
                    best = Some((convergence, owner));
                }
            }
        }
    }
    best.map(|(_, owner)| owner)
}

fn conservatively_fill_lifecycle_gaps(
    mesh: &CarrierMesh,
    mut states: Vec<Option<LifecycleCell>>,
) -> Vec<LifecycleCell> {
    let n = states.len();
    let mut donor = vec![usize::MAX; n];
    let mut queue = std::collections::VecDeque::new();
    for cell in 0..n {
        if states[cell].is_some() {
            donor[cell] = cell;
            queue.push_back(cell);
        }
    }
    while let Some(cell) = queue.pop_front() {
        for &next in &mesh.neighbors[cell] {
            let next = next as usize;
            if donor[next] == usize::MAX {
                donor[next] = donor[cell];
                queue.push_back(next);
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n];
    for cell in 0..n {
        groups[donor[cell]].push(cell);
    }
    let mut output: Vec<Option<LifecycleCell>> = vec![None; n];
    for source in 0..n {
        if groups[source].is_empty() {
            continue;
        }
        let state = states[source].take().expect("gap donor is occupied");
        let total_area: f64 = groups[source]
            .iter()
            .map(|&cell| mesh.areas[cell] as f64)
            .sum();
        for &cell in &groups[source] {
            let mut split = state.clone();
            split.scale_material(mesh.areas[cell] as f64 / total_area.max(1e-30));
            output[cell] = Some(split);
        }
    }
    output
        .into_iter()
        .map(|state| state.expect("all carrier gaps filled"))
        .collect()
}

fn count_lifecycle_sutures(mesh: &CarrierMesh, states: &[LifecycleCell]) -> usize {
    let mut seen = vec![false; states.len()];
    let mut components = 0;
    for start in 0..states.len() {
        if seen[start] || states[start].weakness <= 0.0 {
            continue;
        }
        components += 1;
        seen[start] = true;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(cell) = queue.pop_front() {
            for &next in &mesh.neighbors[cell] {
                let next = next as usize;
                if !seen[next] && states[next].weakness > 0.0 {
                    seen[next] = true;
                    queue.push_back(next);
                }
            }
        }
    }
    components
}

fn distribute_parcel_volume(
    mesh: &CarrierMesh,
    snapshot: &CarrierSnapshot,
    parcel_volume: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n = mesh.centers.len();
    let mut owned_area = vec![0.0f32; n];
    for cell in 0..n {
        owned_area[snapshot.surface_parcel[cell] as usize] += mesh.areas[cell];
    }
    let thickness = (0..n)
        .map(|cell| {
            let parcel = snapshot.surface_parcel[cell] as usize;
            parcel_volume[parcel] / owned_area[parcel].max(1e-12)
        })
        .collect();
    (thickness, owned_area)
}

fn gather_visible_parcel_volume(
    mesh: &CarrierMesh,
    snapshot: &CarrierSnapshot,
    owned_area: &[f32],
    thickness: &[f32],
    parcel_volume: &mut [f32],
) {
    for (parcel, &area) in owned_area.iter().enumerate() {
        if area > 0.0 {
            parcel_volume[parcel] = 0.0;
        }
    }
    for cell in 0..mesh.centers.len() {
        parcel_volume[snapshot.surface_parcel[cell] as usize] += thickness[cell] * mesh.areas[cell];
    }
}

fn carrier_receiver_parcels(mesh: &CarrierMesh, snapshot: &CarrierSnapshot) -> HashSet<u16> {
    build_carrier_boundaries(mesh, snapshot)
        .into_iter()
        .filter(|boundary| boundary.kind == BoundaryKind::Convergent)
        .flat_map(|boundary| match boundary.polarity {
            None if boundary.type_a == CrustType::Continental
                && boundary.type_b == CrustType::Continental =>
            {
                vec![
                    snapshot.surface_parcel[boundary.a],
                    snapshot.surface_parcel[boundary.b],
                ]
            }
            Some(SubductionPolarity::ASubducts) => {
                vec![snapshot.surface_parcel[boundary.b]]
            }
            Some(SubductionPolarity::BSubducts) => {
                vec![snapshot.surface_parcel[boundary.a]]
            }
            _ => Vec::new(),
        })
        .collect()
}

fn carrier_step(mesh: &CarrierMesh, snapshot: &CarrierSnapshot) -> CarrierStep {
    let n = mesh.centers.len();
    let sheet_edges: Vec<_> = mesh
        .edges
        .iter()
        .filter(|edge| {
            snapshot.plate_owner[edge.a] == snapshot.plate_owner[edge.b]
                && snapshot.crust_owner[edge.a] == snapshot.crust_owner[edge.b]
        })
        .map(|edge| SheetEdge {
            a: edge.a,
            b: edge.b,
            conductance: edge.conductance,
            face_length: edge.face_length,
            normal_a_to_b: edge.normal_a_to_b,
        })
        .collect();
    let boundaries = build_carrier_boundaries(mesh, snapshot);
    let boundary_length_km = boundaries
        .iter()
        .map(|boundary| boundary.edge_length * PLANET_RADIUS_KM)
        .sum();
    let convergent_swept_area_km2_per_myr = boundaries
        .iter()
        .filter(|boundary| boundary.kind == BoundaryKind::Convergent)
        .map(|boundary| {
            boundary.convergence.max(0.0)
                * MAX_PLATE_ANGULAR_SPEED_RAD_PER_MYR
                * boundary.edge_length
                * PLANET_RADIUS_KM
                * PLANET_RADIUS_KM
        })
        .sum();
    let mut boundary_cells = HashSet::new();
    for boundary in &boundaries {
        boundary_cells.insert(boundary.a);
        boundary_cells.insert(boundary.b);
    }
    let boundary_support_pct = 100.0 * boundary_cells.len() as f32 / n.max(1) as f32;
    let mut target_sum = vec![Vec3::ZERO; n];
    let mut target_weight = vec![0.0f32; n];
    let mut magmatic_volume_rate = vec![0.0f32; n];
    let mut receiver_parcels = HashSet::new();
    let rate_scale = MAX_PLATE_ANGULAR_SPEED_RAD_PER_MYR;
    for boundary in boundaries {
        if boundary.kind != BoundaryKind::Convergent {
            continue;
        }
        let closing = boundary.convergence.max(0.0) * rate_scale;
        if closing <= 0.0 {
            continue;
        }
        let mut add_target = |cell: usize, magnitude: f32| {
            let center = mesh.centers[cell];
            let toward = boundary.point - center * center.dot(boundary.point);
            if magnitude > 0.0 && toward.length_squared() > 1e-12 {
                target_sum[cell] += -toward.normalize() * magnitude * boundary.edge_length;
                target_weight[cell] += boundary.edge_length;
                receiver_parcels.insert(snapshot.surface_parcel[cell]);
            }
        };
        match boundary.polarity {
            None if boundary.type_a == CrustType::Continental
                && boundary.type_b == CrustType::Continental =>
            {
                add_target(boundary.a, 0.5 * closing);
                add_target(boundary.b, 0.5 * closing);
            }
            Some(SubductionPolarity::ASubducts) => {
                if boundary.type_b == CrustType::Continental {
                    add_target(boundary.b, SUBDUCTION_COMPRESSION_COUPLING * closing);
                } else {
                    receiver_parcels.insert(snapshot.surface_parcel[boundary.b]);
                }
                magmatic_volume_rate[boundary.b] += closing
                    * boundary.edge_length
                    * SUBDUCTION_MAGMATIC_ACCRETION
                    * CRUST_THICKNESS_OCEANIC;
            }
            Some(SubductionPolarity::BSubducts) => {
                if boundary.type_a == CrustType::Continental {
                    add_target(boundary.a, SUBDUCTION_COMPRESSION_COUPLING * closing);
                } else {
                    receiver_parcels.insert(snapshot.surface_parcel[boundary.a]);
                }
                magmatic_volume_rate[boundary.a] += closing
                    * boundary.edge_length
                    * SUBDUCTION_MAGMATIC_ACCRETION
                    * CRUST_THICKNESS_OCEANIC;
            }
            _ => {}
        }
    }
    let target: Vec<_> = target_sum
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
    let coupling = (HISTORY_SHEET_STRESS_TRANSMISSION_KM / PLANET_RADIUS_KM).powi(2);
    let velocity =
        solve_velocity_geometry(&mesh.centers, &mesh.areas, &sheet_edges, &target, coupling);
    let target_l1 = target
        .iter()
        .zip(mesh.areas.iter())
        .map(|(target, &area)| target.length() as f64 * area as f64)
        .sum();
    let magma_source: Vec<_> = magmatic_volume_rate
        .iter()
        .zip(mesh.areas.iter())
        .map(|(&rate, &area)| rate / area.max(1e-12))
        .collect();
    let magmatic_rate = solve_screened_scalar(&mesh.areas, &sheet_edges, &magma_source, coupling);
    let mut strain_rate = vec![0.0f32; n];
    let mut strongest = vec![0.0f32; n];
    let mut compression_axis = vec![Vec3::ZERO; n];
    let mut outgoing_rate = vec![0.0f32; n];
    for edge in &sheet_edges {
        let midpoint = (mesh.centers[edge.a] + mesh.centers[edge.b]).normalize_or_zero();
        let va = velocity[edge.a] - midpoint * midpoint.dot(velocity[edge.a]);
        let vb = velocity[edge.b] - midpoint * midpoint.dot(velocity[edge.b]);
        let speed = (0.5 * (va + vb)).dot(edge.normal_a_to_b);
        let (donor, outward) = if speed > 0.0 {
            (edge.a, speed)
        } else {
            (edge.b, -speed)
        };
        outgoing_rate[donor] += outward * edge.face_length / mesh.areas[donor].max(1e-12);
        let distance = (mesh.centers[edge.b] - mesh.centers[edge.a])
            .length()
            .max(1e-8);
        let normal_strain = (vb - va).dot(edge.normal_a_to_b) / distance;
        strain_rate[edge.a] += 0.5 * normal_strain.abs();
        strain_rate[edge.b] += 0.5 * normal_strain.abs();
        if normal_strain < -strongest[edge.a] {
            strongest[edge.a] = -normal_strain;
            compression_axis[edge.a] = edge.normal_a_to_b;
        }
        if normal_strain < -strongest[edge.b] {
            strongest[edge.b] = -normal_strain;
            compression_axis[edge.b] = edge.normal_a_to_b;
        }
    }
    CarrierStep {
        sheet_edges,
        velocity,
        magmatic_rate,
        magmatic_volume_rate: magmatic_volume_rate.iter().map(|&rate| rate as f64).sum(),
        strain_rate,
        compression_axis,
        outgoing_rate,
        receiver_parcels,
        boundary_length_km,
        convergent_swept_area_km2_per_myr,
        boundary_support_pct,
        target_l1,
    }
}

fn carrier_cell_volume_rate(mesh: &CarrierMesh, thickness: &[f32], step: &CarrierStep) -> Vec<f32> {
    let mut rate = vec![0.0f32; mesh.centers.len()];
    for edge in &step.sheet_edges {
        let midpoint = (mesh.centers[edge.a] + mesh.centers[edge.b]).normalize_or_zero();
        let va = step.velocity[edge.a] - midpoint * midpoint.dot(step.velocity[edge.a]);
        let vb = step.velocity[edge.b] - midpoint * midpoint.dot(step.velocity[edge.b]);
        let speed = (0.5 * (va + vb)).dot(edge.normal_a_to_b);
        let donor_h = if speed >= 0.0 {
            thickness[edge.a]
        } else {
            thickness[edge.b]
        };
        let flux = donor_h * speed * edge.face_length;
        rate[edge.a] -= flux;
        rate[edge.b] += flux;
    }
    for cell in 0..rate.len() {
        rate[cell] += step.magmatic_rate[cell].max(0.0) * mesh.areas[cell];
    }
    rate
}

fn history_duration_myr(replay: &super::history::CarrierReplay) -> f32 {
    replay
        .snapshots
        .last()
        .map(|snapshot| snapshot.lookback_myr)
        .unwrap_or(0.0)
}

fn evolve_frozen_carrier(
    mesh: &CarrierMesh,
    step: &CarrierStep,
    initial: &[f32],
    interval_myr: f32,
    intervals: usize,
    include_transport: bool,
    include_magma: bool,
) -> Vec<f32> {
    let mut thickness = initial.to_vec();
    let max_courant = interval_myr * step.outgoing_rate.iter().copied().fold(0.0f32, f32::max);
    let substeps = (max_courant / 0.20).ceil().max(1.0) as usize;
    let dt = interval_myr / substeps as f32;
    let mut volume_change = vec![0.0f32; mesh.centers.len()];
    for _ in 0..intervals {
        for _ in 0..substeps {
            volume_change.fill(0.0);
            for edge in &step.sheet_edges {
                if !include_transport {
                    continue;
                }
                let midpoint = (mesh.centers[edge.a] + mesh.centers[edge.b]).normalize_or_zero();
                let va = step.velocity[edge.a] - midpoint * midpoint.dot(step.velocity[edge.a]);
                let vb = step.velocity[edge.b] - midpoint * midpoint.dot(step.velocity[edge.b]);
                let speed = (0.5 * (va + vb)).dot(edge.normal_a_to_b);
                let donor_h = if speed >= 0.0 {
                    thickness[edge.a]
                } else {
                    thickness[edge.b]
                };
                let flux = donor_h * speed * edge.face_length;
                volume_change[edge.a] -= dt * flux;
                volume_change[edge.b] += dt * flux;
            }
            for cell in 0..thickness.len() {
                thickness[cell] += volume_change[cell] / mesh.areas[cell].max(1e-12);
                if include_magma {
                    thickness[cell] += dt * step.magmatic_rate[cell].max(0.0);
                }
            }
        }
        let gravity_tau = CRUST_GRAVITATIONAL_DIFFUSIVITY_KM2_PER_MYR
            / (PLANET_RADIUS_KM * PLANET_RADIUS_KM)
            * interval_myr;
        if gravity_tau > 0.0 {
            thickness =
                solve_screened_scalar(&mesh.areas, &step.sheet_edges, &thickness, gravity_tau);
        }
    }
    thickness
}

fn thickness_change_stats(after: &[f32], before: &[f32], areas: &[f32]) -> (f64, f64, f32, f64) {
    let delta: Vec<_> = after
        .iter()
        .zip(before.iter())
        .map(|(&after, &before)| after - before)
        .collect();
    field_stats(&delta, areas)
}

fn field_stats(field: &[f32], areas: &[f32]) -> (f64, f64, f32, f64) {
    let mut positive = 0.0f64;
    let mut negative = 0.0f64;
    let mut max = f32::NEG_INFINITY;
    let mut net = 0.0f64;
    for (&value, &area) in field.iter().zip(areas.iter()) {
        let volume = value as f64 * area as f64;
        positive += volume.max(0.0);
        negative += (-volume).max(0.0);
        max = max.max(value);
        net += volume;
    }
    (positive, negative, max, net)
}

fn canonical_plate_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn classify_carrier_pair(stats: &CarrierPairStats) -> BoundaryKind {
    if stats.length < PLATE_PAIR_MIN_BOUNDARY_LENGTH {
        return BoundaryKind::Transform;
    }
    let normal = stats.normal / stats.length.max(1e-12);
    let shear = stats.shear / stats.length.max(1e-12);
    if shear > normal.abs() * TRANSFORM_RATIO && normal.abs() < TRANSFORM_NORMAL_THRESHOLD {
        BoundaryKind::Transform
    } else if normal > TRANSFORM_NORMAL_THRESHOLD
        && stats.positive_length >= PLATE_PAIR_MIN_ACTIVE_LENGTH
    {
        BoundaryKind::Convergent
    } else if normal < -TRANSFORM_NORMAL_THRESHOLD
        && stats.negative_length >= PLATE_PAIR_MIN_ACTIVE_LENGTH
    {
        BoundaryKind::Divergent
    } else {
        BoundaryKind::Transform
    }
}

fn build_carrier_boundaries(
    mesh: &CarrierMesh,
    snapshot: &CarrierSnapshot,
) -> Vec<CarrierBoundary> {
    struct Raw {
        a: usize,
        b: usize,
        point: Vec3,
        length: f32,
        convergence: f32,
        pair: (usize, usize),
        type_a: CrustType,
        type_b: CrustType,
    }
    let mut raw = Vec::new();
    let mut pair_stats: HashMap<(usize, usize), CarrierPairStats> = HashMap::new();
    for edge in &mesh.edges {
        let plate_a = snapshot.plate_owner[edge.a] as usize;
        let plate_b = snapshot.plate_owner[edge.b] as usize;
        if plate_a == plate_b {
            continue;
        }
        let point = (mesh.centers[edge.a] + mesh.centers[edge.b]).normalize_or_zero();
        let along = point.cross(edge.normal_a_to_b).normalize_or_zero();
        let velocity_a = snapshot.plate_euler_poles[plate_a].velocity_at(point);
        let velocity_b = snapshot.plate_euler_poles[plate_b].velocity_at(point);
        let relative = velocity_a - velocity_b;
        let convergence = relative.dot(edge.normal_a_to_b);
        let shear = relative.dot(along).abs();
        let pair = canonical_plate_pair(plate_a, plate_b);
        let stats = pair_stats.entry(pair).or_default();
        stats.length += edge.face_length;
        stats.normal += convergence * edge.face_length;
        stats.shear += shear * edge.face_length;
        if convergence >= TRANSFORM_NORMAL_THRESHOLD {
            stats.positive_length += edge.face_length;
        } else if convergence <= -TRANSFORM_NORMAL_THRESHOLD {
            stats.negative_length += edge.face_length;
        }
        let type_a = snapshot.crust_owner[edge.a];
        let type_b = snapshot.crust_owner[edge.b];
        if type_a == CrustType::Oceanic && type_b == CrustType::Oceanic && convergence > 0.0 {
            let toward_a = velocity_a.dot(edge.normal_a_to_b).max(0.0);
            let toward_b = velocity_b.dot(-edge.normal_a_to_b).max(0.0);
            let weight = convergence * edge.face_length;
            let (min_toward, max_toward) = if plate_a < plate_b {
                (toward_a, toward_b)
            } else {
                (toward_b, toward_a)
            };
            if min_toward >= max_toward {
                stats.ocean_min_vote += weight;
            } else {
                stats.ocean_max_vote += weight;
            }
        }
        raw.push(Raw {
            a: edge.a,
            b: edge.b,
            point,
            length: edge.face_length,
            convergence,
            pair,
            type_a,
            type_b,
        });
    }
    let kinds: HashMap<_, _> = pair_stats
        .iter()
        .map(|(&pair, stats)| (pair, classify_carrier_pair(stats)))
        .collect();
    raw.into_iter()
        .map(|edge| {
            let kind = kinds[&edge.pair];
            let plate_a = snapshot.plate_owner[edge.a] as usize;
            let polarity = if kind != BoundaryKind::Convergent {
                None
            } else {
                match (edge.type_a, edge.type_b) {
                    (CrustType::Oceanic, CrustType::Continental) => {
                        Some(SubductionPolarity::ASubducts)
                    }
                    (CrustType::Continental, CrustType::Oceanic) => {
                        Some(SubductionPolarity::BSubducts)
                    }
                    (CrustType::Continental, CrustType::Continental) => None,
                    (CrustType::Oceanic, CrustType::Oceanic) => {
                        let stats = &pair_stats[&edge.pair];
                        let min_subducts = stats.ocean_min_vote >= stats.ocean_max_vote;
                        Some(if (plate_a == edge.pair.0) == min_subducts {
                            SubductionPolarity::ASubducts
                        } else {
                            SubductionPolarity::BSubducts
                        })
                    }
                }
            };
            CarrierBoundary {
                a: edge.a,
                b: edge.b,
                point: edge.point,
                edge_length: edge.length,
                convergence: edge.convergence,
                kind,
                polarity,
                type_a: edge.type_a,
                type_b: edge.type_b,
            }
        })
        .collect()
}

fn nearest_carrier_cell(mesh: &CarrierMesh, point: Vec3, start: usize) -> usize {
    let mut cell = start;
    loop {
        let mut best = cell;
        let mut best_dot = point.dot(mesh.centers[cell]);
        for &next in &mesh.neighbors[cell] {
            let next = next as usize;
            let dot = point.dot(mesh.centers[next]);
            if dot > best_dot {
                best = next;
                best_dot = dot;
            }
        }
        if best == cell {
            return cell;
        }
        cell = best;
    }
}

fn project_carrier_scalar(target: &Tessellation, mesh: &CarrierMesh, values: &[f32]) -> Vec<f32> {
    let mut hint = 0usize;
    (0..target.num_cells())
        .map(|cell| {
            hint = nearest_carrier_cell(mesh, target.cell_center(cell), hint);
            values[hint]
        })
        .collect()
}

fn project_carrier_vec3(target: &Tessellation, mesh: &CarrierMesh, values: &[Vec3]) -> Vec<Vec3> {
    let mut hint = 0usize;
    (0..target.num_cells())
        .map(|cell| {
            hint = nearest_carrier_cell(mesh, target.cell_center(cell), hint);
            values[hint].normalize_or_zero()
        })
        .collect()
}

fn project_carrier_bool(target: &Tessellation, mesh: &CarrierMesh, values: &[bool]) -> Vec<bool> {
    let mut hint = 0usize;
    (0..target.num_cells())
        .map(|cell| {
            hint = nearest_carrier_cell(mesh, target.cell_center(cell), hint);
            values[hint]
        })
        .collect()
}

fn solve_velocity_geometry(
    centers: &[Vec3],
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
            let velocity = Vec3::new(x[i], y[i], z[i]);
            velocity - centers[i] * centers[i].dot(velocity)
        })
        .collect()
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
    use crate::world::{Crust, Dynamics, Plates};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

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

    #[test]
    fn coarse_denudation_is_rate_limited_and_spares_reference_crust() {
        let mut thickness = [1.20, 1.00, 0.20];
        let reference = [1.00, 1.00, 0.25];
        let areas = [2.0, 3.0, 5.0];
        let removed = denude_excess_thickness(&mut thickness, &reference, &areas, 0.10, 2.0);
        let isostasy_slope = (CONTINENTAL_BASE - ABYSSAL_DEPTH)
            / (CRUST_THICKNESS_CONTINENTAL - CRUST_THICKNESS_OCEANIC);
        let expected_loss = 0.20 / (ELEVATION_UNIT_KM * isostasy_slope);
        assert!((thickness[0] - (1.20 - expected_loss)).abs() < 1e-6);
        assert_eq!(thickness[1], 1.00);
        assert_eq!(thickness[2], 0.20);
        assert!((removed - expected_loss as f64 * 2.0).abs() < 1e-6);

        let unchanged = thickness;
        assert_eq!(
            denude_excess_thickness(&mut thickness, &reference, &areas, 0.0, 2.0),
            0.0
        );
        assert_eq!(thickness, unchanged);
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

    fn evolved_fixture_with_motion(
        step_myr: f32,
        motion_coherence_myr: f32,
    ) -> (Tessellation, Dynamics, super::super::history::CarrierReplay) {
        let mut mesh_rng = ChaCha8Rng::seed_from_u64(301);
        let tessellation = Tessellation::generate(512, 0, &mut mesh_rng);
        let mut plate_rng = ChaCha8Rng::seed_from_u64(302);
        let plates = Plates::generate(&tessellation, 6, &mut plate_rng);
        let mut crust_rng = ChaCha8Rng::seed_from_u64(303);
        let crust = Crust::generate(&tessellation, 3, 0.3, &mut crust_rng);
        let mut dynamics_rng = ChaCha8Rng::seed_from_u64(304);
        let mut dynamics = Dynamics::generate(&plates, &mut dynamics_rng);
        dynamics.clock.lookback_myr = 8.0;
        let (_, replay) = super::super::history::replay_fixed_carrier(
            305,
            &tessellation,
            &plates,
            &crust,
            &dynamics,
            &HashSet::new(),
            256,
            step_myr,
            false,
            0.0,
            motion_coherence_myr,
        );
        (tessellation, dynamics, replay)
    }

    fn evolved_fixture(
        step_myr: f32,
    ) -> (Tessellation, Dynamics, super::super::history::CarrierReplay) {
        evolved_fixture_with_motion(step_myr, 0.0)
    }

    #[test]
    fn carrier_evolution_is_deterministic_and_mass_conservative() {
        let (tessellation, dynamics, mut replay) = evolved_fixture(2.0);
        replay.operator_audit = true;
        let a = solve_carrier_replay_evolved(&tessellation, &dynamics, &replay, Instant::now());
        let b = solve_carrier_replay_evolved(&tessellation, &dynamics, &replay, Instant::now());
        assert_eq!(a.thickness_delta, b.thickness_delta);
        assert_eq!(a.strain, b.strain);
        assert_eq!(a.compression_axis, b.compression_axis);
        assert_eq!(a.present_uplift_rate, b.present_uplift_rate);
        assert_eq!(a.material_added, b.material_added);
        assert_eq!(a.material_removed, b.material_removed);
        assert_eq!(a.material_residual, b.material_residual);
        assert_eq!(a.operator_audit, b.operator_audit);
        assert!(a.operator_audit.is_some());
        assert!(
            a.material_residual.abs() < 1e-5
                || a.material_residual.abs() / a.material_added.abs().max(1e-30) < 1e-4,
            "mass residual {} for added {}",
            a.material_residual,
            a.material_added,
        );
    }

    #[test]
    fn carrier_denudation_closes_the_material_ledger() {
        let (tessellation, dynamics, mut replay) = evolved_fixture(2.0);
        replay.denudation_rate_km_per_myr = 0.10;
        let result =
            solve_carrier_replay_evolved(&tessellation, &dynamics, &replay, Instant::now());
        assert!(result.material_removed > 0.0);
        let throughput = result.material_added + result.material_removed;
        assert!(
            result.material_residual.abs() < 1e-5
                || result.material_residual.abs() / throughput.max(1e-30) < 1e-4,
            "mass residual {} for ledger throughput {}",
            result.material_residual,
            throughput,
        );
    }

    #[test]
    fn reorganizing_carrier_closes_the_material_ledger() {
        let (tessellation, dynamics, replay) = evolved_fixture_with_motion(2.0, 2.0);
        assert!(replay.mean_reorganizations_per_plate > 0.0);
        let result =
            solve_carrier_replay_evolved(&tessellation, &dynamics, &replay, Instant::now());
        assert!(
            result.material_residual.abs() < 1e-5
                || result.material_residual.abs() / result.material_added.abs().max(1e-30) < 1e-4,
            "mass residual {} for added {}",
            result.material_residual,
            result.material_added,
        );
    }

    #[test]
    fn carrier_evolution_is_stable_under_time_subdivision() {
        let (tessellation, dynamics, replay_1) = evolved_fixture(1.0);
        let (_, _, replay_2) = evolved_fixture(2.0);
        let (_, _, replay_4) = evolved_fixture(4.0);
        let one = solve_carrier_replay_evolved(&tessellation, &dynamics, &replay_1, Instant::now());
        let two = solve_carrier_replay_evolved(&tessellation, &dynamics, &replay_2, Instant::now());
        let four =
            solve_carrier_replay_evolved(&tessellation, &dynamics, &replay_4, Instant::now());
        let relative_rms = |a: &[f32], b: &[f32]| {
            let error = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale = a
                .iter()
                .map(|&x| (x as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            error / scale
        };
        assert!(relative_rms(&one.thickness_delta, &two.thickness_delta) < 0.10);
        assert!(relative_rms(&two.thickness_delta, &four.thickness_delta) < 0.20);
    }

    fn lifecycle_result(step_myr: f32) -> ThinSheetFields {
        let (tessellation, dynamics, replay) = evolved_fixture(step_myr);
        solve_carrier_lifecycle_replay(&tessellation, &dynamics, &replay, 8.0, Instant::now())
    }

    #[test]
    fn lifecycle_is_deterministic_and_closes_all_material_ledgers() {
        let a = lifecycle_result(2.0);
        let b = lifecycle_result(2.0);
        assert_eq!(a.thickness_delta, b.thickness_delta);
        assert_eq!(a.final_continental, b.final_continental);
        assert_eq!(a.final_ocean_age_myr, b.final_ocean_age_myr);
        assert_eq!(a.final_weakness, b.final_weakness);
        let audit_a = a.lifecycle_audit.as_ref().unwrap();
        let audit_b = b.lifecycle_audit.as_ref().unwrap();
        assert_eq!(audit_a.created_ocean_volume, audit_b.created_ocean_volume);
        assert_eq!(audit_a.consumed_ocean_volume, audit_b.consumed_ocean_volume);
        assert_eq!(audit_a.plate_merges, audit_b.plate_merges);
        assert_eq!(audit_a.final_plate_count, audit_b.final_plate_count);
        assert!(audit_a.material_residual.abs() < 1e-10);
        assert!(audit_a.continental_material_residual.abs() < 1e-10);
        assert_eq!(audit_a.final_unresolved_overlaps, 0);
    }

    #[test]
    fn lifecycle_ocean_clock_starts_at_zero_and_advances_in_myr() {
        let mut states = vec![new_lifecycle_ocean(0, 1.0)];
        assert_eq!(states[0].ocean_age_myr, 0.0);
        age_lifecycle_ocean(&mut states, 2.0);
        assert_eq!(states[0].ocean_age_myr, 2.0);
    }

    #[test]
    fn rigid_uniform_plate_is_an_exact_lifecycle_invariant() {
        let (tessellation, dynamics, mut replay) = evolved_fixture(2.0);
        for plate in &mut replay.snapshots[0].plate_owner {
            *plate = 0;
        }
        for crust in &mut replay.snapshots[0].crust_owner {
            *crust = CrustType::Oceanic;
        }
        let result = solve_carrier_lifecycle_replay(
            &tessellation,
            &dynamics,
            &replay,
            100.0,
            Instant::now(),
        );
        let max_delta = result
            .thickness_delta
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max);
        let audit = result.lifecycle_audit.as_ref().unwrap();
        assert!(
            max_delta < 1e-5,
            "rigid rotation made thickness {max_delta}"
        );
        assert_eq!(audit.created_ocean_volume, 0.0);
        assert_eq!(audit.consumed_ocean_volume, 0.0);
        assert_eq!(audit.magmatic_added_volume, 0.0);
        assert_eq!(audit.continental_underthrust_volume, 0.0);
        assert_eq!(audit.plate_merges, 0);
        assert_eq!(audit.material_residual, 0.0);
    }

    #[test]
    fn continental_overlap_creates_suture_without_deleting_mass() {
        let (_, dynamics, replay) = evolved_fixture(2.0);
        let mesh = &replay.mesh;
        let area = mesh.areas[0] as f64;
        let make = |plate| LifecycleCell {
            plate,
            crust: CrustType::Continental,
            ocean_age_myr: 0.0,
            continental_volume: CRUST_THICKNESS_CONTINENTAL as f64 * area,
            ocean_volume: 0.0,
            magma_volume: 0.0,
            continental_area: area,
            ocean_area: 0.0,
            underthrust_volume: 0.0,
            weakness: 0.0,
            fabric: Vec3::ZERO,
            collision_deposits: 0,
        };
        let states = vec![make(0), make(1)];
        let mut parent: Vec<_> = (0..dynamics.euler_poles.len()).collect();
        let mut speeds = HashMap::new();
        let closure_rates = HashMap::from([((0, 1), 50.0)]);
        let mut audit = LifecycleAudit::default();
        let result = resolve_lifecycle_overlap(
            0,
            &[0, 1],
            &states,
            mesh,
            &dynamics.euler_poles,
            &mut parent,
            &closure_rates,
            &mut speeds,
            &mut audit,
        );
        assert_eq!(
            result.continental_volume,
            states[0].continental_volume * 2.0
        );
        assert_eq!(result.weakness, 1.0);
        assert_eq!(
            audit.continental_underthrust_volume,
            states[1].continental_volume
        );
        assert_eq!(audit.consumed_ocean_volume, 0.0);
        assert!(!speeds.is_empty());
    }

    #[test]
    fn oceanic_overlap_cannot_trigger_plate_merge() {
        let (_, dynamics, replay) = evolved_fixture(2.0);
        let mesh = &replay.mesh;
        let area = mesh.areas[0] as f64;
        let make = |plate| LifecycleCell {
            plate,
            crust: CrustType::Oceanic,
            ocean_age_myr: 10.0,
            continental_volume: 0.0,
            ocean_volume: CRUST_THICKNESS_OCEANIC as f64 * area,
            magma_volume: 0.0,
            continental_area: 0.0,
            ocean_area: area,
            underthrust_volume: 0.0,
            weakness: 0.0,
            fabric: Vec3::ZERO,
            collision_deposits: 0,
        };
        let states = vec![make(0), make(1)];
        let mut parent: Vec<_> = (0..dynamics.euler_poles.len()).collect();
        let mut speeds = HashMap::new();
        let closure_rates = HashMap::new();
        let mut audit = LifecycleAudit::default();
        let _ = resolve_lifecycle_overlap(
            0,
            &[0, 1],
            &states,
            mesh,
            &dynamics.euler_poles,
            &mut parent,
            &closure_rates,
            &mut speeds,
            &mut audit,
        );
        assert!(speeds.is_empty());
        assert_eq!(parent, (0..dynamics.euler_poles.len()).collect::<Vec<_>>());
    }

    #[test]
    fn continental_merge_clock_uses_normal_convergence_not_transform_speed() {
        let (_, _, replay) = evolved_fixture(2.0);
        let mesh = &replay.mesh;
        let edge = mesh.edges[0];
        let mut states: Vec<_> = (0..mesh.centers.len())
            .map(|cell| new_lifecycle_ocean(0, mesh.areas[cell] as f64))
            .collect();
        for (cell, plate) in [(edge.a, 0), (edge.b, 1)] {
            states[cell].plate = plate;
            states[cell].crust = CrustType::Continental;
            states[cell].continental_volume =
                CRUST_THICKNESS_CONTINENTAL as f64 * mesh.areas[cell] as f64;
            states[cell].ocean_volume = 0.0;
            states[cell].continental_area = mesh.areas[cell] as f64;
            states[cell].ocean_area = 0.0;
        }
        let point = (mesh.centers[edge.a] + mesh.centers[edge.b]).normalize();
        let tangent = point.cross(edge.normal_a_to_b).normalize();
        let pole_for_velocity = |velocity: Vec3| EulerPole {
            axis: point.cross(velocity).normalize(),
            angular_velocity: 1.0,
        };
        let stationary = EulerPole {
            axis: Vec3::Z,
            angular_velocity: 0.0,
        };
        let transform = continental_pair_closure_rates(
            mesh,
            &states,
            &[pole_for_velocity(tangent), stationary.clone()],
        );
        assert!(!transform.contains_key(&(0, 1)));
        let convergent = continental_pair_closure_rates(
            mesh,
            &states,
            &[pole_for_velocity(edge.normal_a_to_b), stationary],
        );
        assert!(convergent.get(&(0, 1)).copied().unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn isolated_suture_survives_rigid_pullback_and_time_subdivision() {
        let (_, dynamics, replay) = evolved_fixture(2.0);
        let mesh = &replay.mesh;
        let run = |dt: f32| {
            let mut states: Vec<_> = (0..mesh.centers.len())
                .map(|cell| new_lifecycle_ocean(0, mesh.areas[cell] as f64))
                .collect();
            states[0].weakness = 1.0;
            states[0].collision_deposits = 3;
            states[0].fabric = mesh.centers[0].cross(Vec3::Y).normalize_or_zero();
            let mut parent = vec![0usize; dynamics.euler_poles.len()];
            let poles = dynamics.euler_poles.clone();
            for _ in 0..(8.0 / dt) as usize {
                let (candidates, admissions) =
                    lifecycle_pullback_admissions(mesh, &states, &poles, &mut parent, dt);
                states = admissions
                    .iter()
                    .map(|ids| {
                        assert_eq!(ids.len(), 1);
                        candidates[ids[0]].clone()
                    })
                    .collect();
            }
            (
                states
                    .iter()
                    .map(|state| state.weakness)
                    .fold(0.0f32, f32::max),
                states
                    .iter()
                    .map(|state| state.collision_deposits)
                    .max()
                    .unwrap(),
                states
                    .iter()
                    .filter(|state| state.weakness > 0.0)
                    .map(|state| state.fabric.length())
                    .fold(0.0f32, f32::max),
                states
                    .iter()
                    .enumerate()
                    .filter(|(_, state)| state.weakness > 0.0)
                    .max_by(|(a, state_a), (b, state_b)| {
                        state_a
                            .weakness
                            .total_cmp(&state_b.weakness)
                            .then_with(|| b.cmp(a))
                    })
                    .map(|(cell, _)| mesh.centers[cell])
                    .unwrap(),
            )
        };
        let one = run(1.0);
        let two = run(2.0);
        let four = run(4.0);
        assert_eq!(one.0, 1.0);
        assert_eq!(two.0, 1.0);
        assert_eq!(four.0, 1.0);
        assert_eq!(one.1, 3);
        assert_eq!(two.1, 3);
        assert_eq!(four.1, 3);
        assert!((one.2 - 1.0).abs() < 1e-5);
        assert!((two.2 - 1.0).abs() < 1e-5);
        assert!((four.2 - 1.0).abs() < 1e-5);
        let expected = Quat::from_axis_angle(
            dynamics.euler_poles[0].axis,
            dynamics.euler_poles[0].angular_velocity_rad_per_myr() * 8.0,
        ) * mesh.centers[0];
        let tolerance = 2.0 * replay.mean_spacing_km / PLANET_RADIUS_KM;
        for position in [one.3, two.3, four.3] {
            assert!(position.dot(expected).clamp(-1.0, 1.0).acos() <= tolerance);
        }
        assert!(one.3.dot(two.3).clamp(-1.0, 1.0).acos() <= tolerance);
        assert!(two.3.dot(four.3).clamp(-1.0, 1.0).acos() <= tolerance);
        assert_eq!(run(2.0), two);
    }

    #[test]
    fn lifecycle_final_state_is_bounded_under_time_subdivision() {
        let one = lifecycle_result(1.0);
        let two = lifecycle_result(2.0);
        let four = lifecycle_result(4.0);
        let agreement = |a: &ThinSheetFields, b: &ThinSheetFields| {
            let a = a.final_continental.as_ref().unwrap();
            let b = b.final_continental.as_ref().unwrap();
            a.iter().zip(b).filter(|(x, y)| x == y).count() as f32 / a.len() as f32
        };
        assert!(agreement(&one, &two) >= 0.70);
        assert!(agreement(&two, &four) >= 0.65);
        for result in [&one, &two, &four] {
            let audit = result.lifecycle_audit.as_ref().unwrap();
            assert!(audit.material_residual.abs() < 1e-10);
            assert!(audit.continental_material_residual.abs() < 1e-10);
            assert_eq!(audit.final_unresolved_overlaps, 0);
        }
    }
}
