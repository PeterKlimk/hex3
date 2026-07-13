//! Registered polygon-mean inputs and local-rank precheck for channel R1a.
//!
//! This module constructs immutable routed cases. It deliberately stops before
//! tracing P0 or M0 paths: a full-domain winner conflict is only a necessary
//! precheck for the later visited-cell discriminator.

use std::{cmp::Ordering, fmt};

use glam::{DVec2, DVec3};

use super::{
    BoundaryFaceCondition, FaceFlowCache, FlowPartition, OutletPortalId, VoronoiCapFixture,
    R1_CAP_PORTAL_ID,
};

pub const R1_RUNOFF_DEPTH_RATE_KM_MYR: f64 = 500.0;
pub const R1_HEAD_ALONG_TRACK_KM: f64 = 176.0;
pub const R1_BROAD_HALF_WIDTH_KM: f64 = 12.0;
pub const R1_HEAD_BOUNDARY_CROSS_TOLERANCE_FACTOR: f64 = 1.0e-10;

const R1_BASE_ELEVATION_KM: f64 = 1.0;
const R1_ALONG_TRACK_GRADE: f64 = 0.01;
const R1_TRANSVERSE_CURVATURE_PER_KM: f64 = 0.0008;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R1SurfaceKind {
    Affine,
    Valley,
    Broad,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct R1CaseConfig {
    pub spacing_km: f64,
    pub theta_rad: f64,
    pub outlet_offset_km: f64,
    pub surface: R1SurfaceKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1HeadOwner {
    pub transverse_offset_km: f64,
    pub point_km: DVec2,
    pub cell: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R1SelectedFace {
    Internal {
        directed_edge: usize,
        neighbor: usize,
    },
    Portal {
        boundary_face: usize,
        portal_id: OutletPortalId,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1LocalRank {
    pub face: R1SelectedFace,
    pub score: f64,
    pub second_score: f64,
    pub normalized_margin: f64,
    pub tie_decision: R1RankTieDecision,
    pub midpoint_km: DVec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R1RankTieDecision {
    SoleEligibleFace,
    Score,
    MidpointX,
    MidpointY,
    MidpointZ,
    PortalKey,
    CombinedFaceIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1LocalRankObservation {
    pub p0_physical_grade: R1LocalRank,
    pub m0_mfd_fraction: R1LocalRank,
    pub winners_conflict: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1CaseAudit {
    pub donor_cells: usize,
    pub domain_rank_conflicts: usize,
    pub head_rank_conflicts: usize,
    pub minimum_p0_normalized_margin: f64,
    pub minimum_m0_normalized_margin: f64,
    pub maximum_fraction_sum_error: f64,
    pub maximum_cell_balance_error_km3_myr: f64,
    pub water_balance_error_km3_myr: f64,
    pub water_balance_relative_error: f64,
    pub total_supply_km3_myr: f64,
    pub target_portal_outflow_km3_myr: f64,
    pub total_sink_storage_km3_myr: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct R1RegisteredCase {
    pub config: R1CaseConfig,
    pub elevation_km: Vec<f64>,
    pub local_supply_km3_myr: Vec<f64>,
    pub flow: FaceFlowCache,
    pub heads: Vec<R1HeadOwner>,
    pub local_ranks: Vec<Option<R1LocalRankObservation>>,
    pub audit: R1CaseAudit,
}

/// Test-only counterfactual whose affine state is sampled at mesh generators.
///
/// The wrapper prevents this deliberately lower-fidelity input from being
/// mistaken for a registered exact polygon-mean case at a call site.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct R1AffineGeneratorPointControl {
    pub(super) case: R1RegisteredCase,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R1CaseError(pub String);

impl fmt::Display for R1CaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for R1CaseError {}

#[derive(Debug, Clone, Copy)]
struct RawPolygonMoments {
    area: f64,
    integral_x: f64,
    integral_y: f64,
    integral_x2: f64,
    integral_y2: f64,
    integral_xy: f64,
}

#[derive(Debug, Clone)]
struct FaceCandidate {
    face: R1SelectedFace,
    midpoint_km: DVec3,
    portal_key: u32,
    combined_face_index: usize,
    physical_grade: f64,
    mfd_fraction: f64,
}

/// Build one registered surface and its one shared immutable MFD route.
pub fn build_r1_registered_case(
    cap: &VoronoiCapFixture,
    config: R1CaseConfig,
) -> Result<R1RegisteredCase, R1CaseError> {
    validate_case_config(cap, config)?;
    let (outlet, along, transverse) = case_frame(config);
    let mut elevation_km = Vec::with_capacity(cap.mesh.cell_count());
    let convexity_tolerance = 1.0e-12 * config.spacing_km.powi(2);
    for (cell, polygon) in cap.cell_polygons_km.iter().enumerate() {
        validate_convex_polygon(polygon, convexity_tolerance).map_err(|error| {
            R1CaseError(format!(
                "projected retained cell {cell} is not convex: {error}"
            ))
        })?;
        let transformed: Vec<_> = polygon
            .iter()
            .map(|point| {
                let relative = *point - outlet;
                DVec2::new(relative.dot(along), relative.dot(transverse))
            })
            .collect();
        let moments = polygon_moments(&transformed)?;
        let area_error = (moments.area - cap.mesh.cell_area_km2[cell]).abs();
        if area_error > 1.0e-10 * cap.mesh.cell_area_km2[cell].max(1.0) {
            return Err(R1CaseError(format!(
                "cell {cell} moment area disagrees with G0 polygon area by {area_error:e} km^2"
            )));
        }
        elevation_km.push(surface_mean(config.surface, &transformed, moments)?);
    }

    assemble_r1_registered_case(cap, config, elevation_km)
}

/// Build the preregistered affine generator-point counterfactual.
///
/// This seam exists only to test whether the exact polygon-mean state is the
/// source of an observed path response. It is not a selectable product input.
#[cfg(test)]
pub(super) fn build_r1_affine_generator_point_control(
    cap: &VoronoiCapFixture,
    config: R1CaseConfig,
) -> Result<R1AffineGeneratorPointControl, R1CaseError> {
    validate_case_config(cap, config)?;
    if config.surface != R1SurfaceKind::Affine {
        return Err(R1CaseError(
            "generator-point control is defined only for the affine surface".into(),
        ));
    }
    let (outlet, along, _) = case_frame(config);
    let elevation_km = cap
        .mesh
        .cell_center_km
        .iter()
        .map(|center| {
            let point = DVec2::new(center.x, center.y);
            let s = (point - outlet).dot(along);
            R1_BASE_ELEVATION_KM + R1_ALONG_TRACK_GRADE * s
        })
        .collect();
    Ok(R1AffineGeneratorPointControl {
        case: assemble_r1_registered_case(cap, config, elevation_km)?,
    })
}

fn assemble_r1_registered_case(
    cap: &VoronoiCapFixture,
    config: R1CaseConfig,
    elevation_km: Vec<f64>,
) -> Result<R1RegisteredCase, R1CaseError> {
    if elevation_km.len() != cap.mesh.cell_count()
        || elevation_km.iter().any(|elevation| !elevation.is_finite())
    {
        return Err(R1CaseError(
            "assembled R1 elevation must contain one finite value per cell".into(),
        ));
    }
    let (outlet, along, transverse) = case_frame(config);

    let head_offsets: &[f64] = match config.surface {
        R1SurfaceKind::Affine | R1SurfaceKind::Valley => &[0.0],
        R1SurfaceKind::Broad => &[-8.0, 8.0],
    };
    let boundary_tolerance = R1_HEAD_BOUNDARY_CROSS_TOLERANCE_FACTOR * config.spacing_km.powi(2);
    let heads = head_offsets
        .iter()
        .map(|&offset| {
            let point = outlet + R1_HEAD_ALONG_TRACK_KM * along + offset * transverse;
            let cell = locate_unique_head(&cap.cell_polygons_km, point, boundary_tolerance)?;
            Ok(R1HeadOwner {
                transverse_offset_km: offset,
                point_km: point,
                cell,
            })
        })
        .collect::<Result<Vec<_>, R1CaseError>>()?;

    let local_supply_km3_myr: Vec<_> = cap
        .mesh
        .cell_area_km2
        .iter()
        .map(|area| R1_RUNOFF_DEPTH_RATE_KM_MYR * area)
        .collect();
    let flow = FaceFlowCache::route_with_portals(
        &cap.mesh,
        &elevation_km,
        &local_supply_km3_myr,
        FlowPartition::MfdSlope,
    )
    .map_err(|error| R1CaseError(error.to_string()))?;
    if flow.routing_elevation_km != elevation_km
        || flow.local_supply_km3_myr != local_supply_km3_myr
        || flow.flat_potential.iter().any(Option::is_some)
    {
        return Err(R1CaseError(
            "unfilled route did not preserve its immutable physical inputs".into(),
        ));
    }

    let boundary_by_cell = boundary_faces_by_cell(cap);
    let reverse_edge = reciprocal_edges(cap)?;
    let mut local_ranks = Vec::with_capacity(cap.mesh.cell_count());
    let mut donor_cells = 0usize;
    let mut domain_rank_conflicts = 0usize;
    let mut minimum_p0_normalized_margin = f64::INFINITY;
    let mut minimum_m0_normalized_margin = f64::INFINITY;
    for (cell, cell_boundary_faces) in boundary_by_cell
        .iter()
        .enumerate()
        .take(cap.mesh.cell_count())
    {
        let candidates = face_candidates(cap, &flow, &elevation_km, cell_boundary_faces, cell)?;
        if candidates.is_empty() {
            local_ranks.push(None);
            continue;
        }
        donor_cells += 1;
        let p0 = rank_candidates(&candidates, |candidate| candidate.physical_grade);
        let m0 = rank_candidates(&candidates, |candidate| candidate.mfd_fraction);
        minimum_p0_normalized_margin = minimum_p0_normalized_margin.min(p0.normalized_margin);
        minimum_m0_normalized_margin = minimum_m0_normalized_margin.min(m0.normalized_margin);
        let winners_conflict = p0.face != m0.face;
        domain_rank_conflicts += usize::from(winners_conflict);
        local_ranks.push(Some(R1LocalRankObservation {
            p0_physical_grade: p0,
            m0_mfd_fraction: m0,
            winners_conflict,
        }));
    }
    let head_rank_conflicts = heads
        .iter()
        .filter(|head| {
            local_ranks[head.cell]
                .as_ref()
                .is_some_and(|rank| rank.winners_conflict)
        })
        .count();
    let route_audit = audit_route(cap, &flow, &boundary_by_cell, &reverse_edge)?;
    let audit = R1CaseAudit {
        donor_cells,
        domain_rank_conflicts,
        head_rank_conflicts,
        minimum_p0_normalized_margin,
        minimum_m0_normalized_margin,
        maximum_fraction_sum_error: route_audit.maximum_fraction_sum_error,
        maximum_cell_balance_error_km3_myr: route_audit.maximum_cell_balance_error_km3_myr,
        water_balance_error_km3_myr: flow.water_balance_error_km3_myr(),
        water_balance_relative_error: flow.water_balance_error_km3_myr().abs()
            / flow.total_supply_km3_myr.max(f64::MIN_POSITIVE),
        total_supply_km3_myr: flow.total_supply_km3_myr,
        target_portal_outflow_km3_myr: flow
            .portal_outflow_km3_myr
            .iter()
            .find(|(portal, _)| *portal == R1_CAP_PORTAL_ID)
            .map(|(_, flux)| *flux)
            .ok_or_else(|| R1CaseError("registered target portal is absent".into()))?,
        total_sink_storage_km3_myr: flow.total_sink_storage_km3_myr,
    };
    if audit.target_portal_outflow_km3_myr <= 0.0 {
        return Err(R1CaseError(
            "registered target portal receives no water".into(),
        ));
    }
    Ok(R1RegisteredCase {
        config,
        elevation_km,
        local_supply_km3_myr,
        flow,
        heads,
        local_ranks,
        audit,
    })
}

fn validate_case_config(cap: &VoronoiCapFixture, config: R1CaseConfig) -> Result<(), R1CaseError> {
    if config.spacing_km != cap.config.spacing_km {
        return Err(R1CaseError(format!(
            "case spacing {} km does not match cap spacing {} km",
            config.spacing_km, cap.config.spacing_km
        )));
    }
    if cap.config.guard_spacings != 8 {
        return Err(R1CaseError(
            "registered routed cases require the promoted eight-spacing guard".into(),
        ));
    }
    if ![8.0, 4.0, 2.0].contains(&config.spacing_km)
        || ![0.0, 0.31].contains(&config.theta_rad)
        || ![0.0, 0.7].contains(&config.outlet_offset_km)
    {
        return Err(R1CaseError(
            "case is outside the registered spacing/angle/translation matrix".into(),
        ));
    }
    Ok(())
}

fn case_frame(config: R1CaseConfig) -> (DVec2, DVec2, DVec2) {
    let along = DVec2::new(config.theta_rad.sin(), config.theta_rad.cos());
    let transverse = DVec2::new(-config.theta_rad.cos(), config.theta_rad.sin());
    (
        DVec2::new(config.outlet_offset_km, -112.0),
        along,
        transverse,
    )
}

fn surface_mean(
    surface: R1SurfaceKind,
    polygon_sn: &[DVec2],
    moments: RawPolygonMoments,
) -> Result<f64, R1CaseError> {
    let mean_s = moments.integral_x / moments.area;
    let transverse_term = match surface {
        R1SurfaceKind::Affine => 0.0,
        R1SurfaceKind::Valley => moments.integral_y2 / moments.area,
        R1SurfaceKind::Broad => broad_exterior_integral(polygon_sn)? / moments.area,
    };
    Ok(R1_BASE_ELEVATION_KM
        + R1_ALONG_TRACK_GRADE * mean_s
        + R1_TRANSVERSE_CURVATURE_PER_KM * transverse_term)
}

fn polygon_moments(vertices: &[DVec2]) -> Result<RawPolygonMoments, R1CaseError> {
    if vertices.len() < 3 {
        return Err(R1CaseError("polygon has fewer than three vertices".into()));
    }
    let mut cross_sum = 0.0;
    let mut integral_x = 0.0;
    let mut integral_y = 0.0;
    let mut integral_x2 = 0.0;
    let mut integral_y2 = 0.0;
    let mut integral_xy = 0.0;
    for (a, b) in vertices
        .iter()
        .copied()
        .zip(vertices.iter().copied().cycle().skip(1))
        .take(vertices.len())
    {
        let cross = a.perp_dot(b);
        cross_sum += cross;
        integral_x += (a.x + b.x) * cross / 6.0;
        integral_y += (a.y + b.y) * cross / 6.0;
        integral_x2 += (a.x * a.x + a.x * b.x + b.x * b.x) * cross / 12.0;
        integral_y2 += (a.y * a.y + a.y * b.y + b.y * b.y) * cross / 12.0;
        integral_xy += (2.0 * a.x * a.y + a.x * b.y + b.x * a.y + 2.0 * b.x * b.y) * cross / 24.0;
    }
    let signed_area = 0.5 * cross_sum;
    if !signed_area.is_finite() || signed_area.abs() <= f64::MIN_POSITIVE {
        return Err(R1CaseError("polygon has invalid signed area".into()));
    }
    let sign = signed_area.signum();
    let moments = RawPolygonMoments {
        area: signed_area.abs(),
        integral_x: integral_x * sign,
        integral_y: integral_y * sign,
        integral_x2: integral_x2 * sign,
        integral_y2: integral_y2 * sign,
        integral_xy: integral_xy * sign,
    };
    if [
        moments.area,
        moments.integral_x,
        moments.integral_y,
        moments.integral_x2,
        moments.integral_y2,
        moments.integral_xy,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(R1CaseError("polygon moments are non-finite".into()));
    }
    Ok(moments)
}

fn broad_exterior_integral(polygon_sn: &[DVec2]) -> Result<f64, R1CaseError> {
    let upper = clip_at_n(polygon_sn, R1_BROAD_HALF_WIDTH_KM, true);
    let lower = clip_at_n(polygon_sn, -R1_BROAD_HALF_WIDTH_KM, false);
    let upper_integral = if upper.len() >= 3 {
        let moments = polygon_moments(&upper)?;
        moments.integral_y2 - 2.0 * R1_BROAD_HALF_WIDTH_KM * moments.integral_y
            + R1_BROAD_HALF_WIDTH_KM.powi(2) * moments.area
    } else {
        0.0
    };
    let lower_integral = if lower.len() >= 3 {
        let moments = polygon_moments(&lower)?;
        moments.integral_y2
            + 2.0 * R1_BROAD_HALF_WIDTH_KM * moments.integral_y
            + R1_BROAD_HALF_WIDTH_KM.powi(2) * moments.area
    } else {
        0.0
    };
    Ok(upper_integral + lower_integral)
}

fn clip_at_n(vertices: &[DVec2], threshold: f64, keep_above: bool) -> Vec<DVec2> {
    let mut clipped = Vec::new();
    let inside = |point: DVec2| {
        if keep_above {
            point.y >= threshold
        } else {
            point.y <= threshold
        }
    };
    for (a, b) in vertices
        .iter()
        .copied()
        .zip(vertices.iter().copied().cycle().skip(1))
        .take(vertices.len())
    {
        let a_inside = inside(a);
        let b_inside = inside(b);
        match (a_inside, b_inside) {
            (true, true) => clipped.push(b),
            (true, false) => clipped.push(intersection_at_n(a, b, threshold)),
            (false, true) => {
                clipped.push(intersection_at_n(a, b, threshold));
                clipped.push(b);
            }
            (false, false) => {}
        }
    }
    clipped
}

fn intersection_at_n(a: DVec2, b: DVec2, threshold: f64) -> DVec2 {
    let fraction = (threshold - a.y) / (b.y - a.y);
    DVec2::new(a.x + fraction * (b.x - a.x), threshold)
}

fn locate_unique_head(
    polygons: &[Vec<DVec2>],
    point: DVec2,
    boundary_tolerance: f64,
) -> Result<usize, R1CaseError> {
    let mut owner = None;
    for (cell, polygon) in polygons.iter().enumerate() {
        match convex_point_relation(polygon, point, boundary_tolerance) {
            PointRelation::Outside => {}
            PointRelation::Boundary => {
                return Err(R1CaseError(format!(
                    "registered head {point:?} is boundary-adjacent at cell {cell}"
                )));
            }
            PointRelation::Inside => {
                if let Some(previous) = owner {
                    return Err(R1CaseError(format!(
                        "registered head {point:?} belongs to cells {previous} and {cell}"
                    )));
                }
                owner = Some(cell);
            }
        }
    }
    owner.ok_or_else(|| R1CaseError(format!("registered head {point:?} has no retained owner")))
}

fn validate_convex_polygon(vertices: &[DVec2], tolerance: f64) -> Result<(), R1CaseError> {
    if vertices.len() < 3 {
        return Err(R1CaseError("fewer than three vertices".into()));
    }
    let mut positive = false;
    let mut negative = false;
    for index in 0..vertices.len() {
        let a = vertices[index];
        let b = vertices[(index + 1) % vertices.len()];
        let c = vertices[(index + 2) % vertices.len()];
        let cross = (b - a).perp_dot(c - b);
        if cross > tolerance {
            positive = true;
        } else if cross < -tolerance {
            negative = true;
        }
        if positive && negative {
            return Err(R1CaseError("successive turns change sign".into()));
        }
    }
    if !positive && !negative {
        return Err(R1CaseError("all successive turns are degenerate".into()));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PointRelation {
    Outside,
    Boundary,
    Inside,
}

fn convex_point_relation(
    polygon: &[DVec2],
    point: DVec2,
    boundary_tolerance: f64,
) -> PointRelation {
    let mut positive = false;
    let mut negative = false;
    let mut boundary = false;
    for (a, b) in polygon
        .iter()
        .copied()
        .zip(polygon.iter().copied().cycle().skip(1))
        .take(polygon.len())
    {
        let cross = (b - a).perp_dot(point - a);
        if cross > boundary_tolerance {
            positive = true;
        } else if cross < -boundary_tolerance {
            negative = true;
        } else {
            boundary = true;
        }
        if positive && negative {
            return PointRelation::Outside;
        }
    }
    if boundary {
        PointRelation::Boundary
    } else {
        PointRelation::Inside
    }
}

fn boundary_faces_by_cell(cap: &VoronoiCapFixture) -> Vec<Vec<usize>> {
    let mut by_cell = vec![Vec::new(); cap.mesh.cell_count()];
    for (index, face) in cap.mesh.boundary_faces.iter().enumerate() {
        by_cell[face.cell as usize].push(index);
    }
    by_cell
}

fn reciprocal_edges(cap: &VoronoiCapFixture) -> Result<Vec<usize>, R1CaseError> {
    let mut reverse = vec![usize::MAX; cap.mesh.edge_neighbor.len()];
    for cell in 0..cap.mesh.cell_count() {
        let start = cap.mesh.edge_offsets[cell] as usize;
        let end = cap.mesh.edge_offsets[cell + 1] as usize;
        for (edge, slot) in reverse.iter_mut().enumerate().take(end).skip(start) {
            let neighbor = cap.mesh.edge_neighbor[edge] as usize;
            *slot = (cap.mesh.edge_offsets[neighbor] as usize
                ..cap.mesh.edge_offsets[neighbor + 1] as usize)
                .find(|&candidate| cap.mesh.edge_neighbor[candidate] as usize == cell)
                .ok_or_else(|| R1CaseError(format!("edge {edge} has no reciprocal")))?;
        }
    }
    Ok(reverse)
}

fn face_candidates(
    cap: &VoronoiCapFixture,
    flow: &FaceFlowCache,
    elevation_km: &[f64],
    boundary_faces: &[usize],
    cell: usize,
) -> Result<Vec<FaceCandidate>, R1CaseError> {
    let mut candidates = Vec::new();
    let start = cap.mesh.edge_offsets[cell] as usize;
    let end = cap.mesh.edge_offsets[cell + 1] as usize;
    for edge in start..end {
        let neighbor = cap.mesh.edge_neighbor[edge] as usize;
        let grade = (elevation_km[cell] - elevation_km[neighbor])
            / f64::from(cap.mesh.edge_distance_km[edge]);
        let fraction = flow.directed_edge_fraction[edge];
        if (grade > 0.0) != (fraction > 0.0) {
            return Err(R1CaseError(format!(
                "cell {cell} edge {edge} grade/fraction eligibility differs"
            )));
        }
        if grade > 0.0 {
            candidates.push(FaceCandidate {
                face: R1SelectedFace::Internal {
                    directed_edge: edge,
                    neighbor,
                },
                midpoint_km: cap.edge_face_midpoint_km[edge],
                portal_key: u32::MAX,
                combined_face_index: edge,
                physical_grade: grade,
                mfd_fraction: fraction,
            });
        }
    }
    for &face_index in boundary_faces {
        let face = &cap.mesh.boundary_faces[face_index];
        let fraction = flow.boundary_face_fraction[face_index];
        match face.condition {
            BoundaryFaceCondition::Closed => {
                if fraction != 0.0 || flow.boundary_face_flux_km3_myr[face_index] != 0.0 {
                    return Err(R1CaseError(format!(
                        "closed boundary face {face_index} carries water"
                    )));
                }
            }
            BoundaryFaceCondition::OpenBaseLevel {
                portal_id,
                elevation_km: base,
            } => {
                let grade = (elevation_km[cell] - f64::from(base)) / face.center_distance_km;
                if (grade > 0.0) != (fraction > 0.0) {
                    return Err(R1CaseError(format!(
                        "cell {cell} portal face {face_index} grade/fraction eligibility differs"
                    )));
                }
                if portal_id != R1_CAP_PORTAL_ID {
                    if fraction > 0.0 || flow.boundary_face_flux_km3_myr[face_index] > 0.0 {
                        return Err(R1CaseError(format!(
                            "wrong portal {portal_id:?} carries water"
                        )));
                    }
                    continue;
                }
                if grade > 0.0 {
                    candidates.push(FaceCandidate {
                        face: R1SelectedFace::Portal {
                            boundary_face: face_index,
                            portal_id,
                        },
                        midpoint_km: face.center_km,
                        portal_key: portal_id.0,
                        combined_face_index: cap.mesh.edge_neighbor.len() + face_index,
                        physical_grade: grade,
                        mfd_fraction: fraction,
                    });
                }
            }
        }
    }
    Ok(candidates)
}

/// Recompute both registered local ranks from only one visited cell's faces.
pub(super) fn rank_r1_case_cell(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    boundary_faces: &[usize],
    cell: usize,
) -> Result<Option<R1LocalRankObservation>, R1CaseError> {
    if cell >= cap.mesh.cell_count()
        || case.elevation_km.len() != cap.mesh.cell_count()
        || case.flow.directed_edge_fraction.len() != cap.mesh.edge_neighbor.len()
        || case.flow.boundary_face_fraction.len() != cap.mesh.boundary_faces.len()
    {
        return Err(R1CaseError(
            "routed case geometry is incompatible with the tracing cap".into(),
        ));
    }
    if boundary_faces.iter().any(|&face_index| {
        cap.mesh
            .boundary_faces
            .get(face_index)
            .is_none_or(|face| face.cell as usize != cell)
    }) {
        return Err(R1CaseError(format!(
            "cell {cell} tracing context contains a foreign boundary face"
        )));
    }
    let candidates = face_candidates(cap, &case.flow, &case.elevation_km, boundary_faces, cell)?;
    if candidates.is_empty() {
        return Ok(None);
    }
    let p0 = rank_candidates(&candidates, |candidate| candidate.physical_grade);
    let m0 = rank_candidates(&candidates, |candidate| candidate.mfd_fraction);
    Ok(Some(R1LocalRankObservation {
        winners_conflict: p0.face != m0.face,
        p0_physical_grade: p0,
        m0_mfd_fraction: m0,
    }))
}

fn rank_candidates(
    candidates: &[FaceCandidate],
    score: impl Fn(&FaceCandidate) -> f64,
) -> R1LocalRank {
    debug_assert!(!candidates.is_empty());
    let preference = |a: usize, b: usize| {
        score(&candidates[b])
            .total_cmp(&score(&candidates[a]))
            .then_with(|| face_key_cmp(&candidates[a], &candidates[b]))
    };
    let mut best = 0usize;
    let mut second = None;
    for candidate in 1..candidates.len() {
        if preference(candidate, best) == Ordering::Less {
            second = Some(best);
            best = candidate;
        } else if second.is_none_or(|runner_up| preference(candidate, runner_up) == Ordering::Less)
        {
            second = Some(candidate);
        }
    }
    let best_candidate = &candidates[best];
    let best_score = score(best_candidate);
    let second_score = second.map_or(0.0, |index| score(&candidates[index]));
    R1LocalRank {
        face: best_candidate.face,
        score: best_score,
        second_score,
        normalized_margin: (best_score - second_score) / best_score.abs().max(f64::MIN_POSITIVE),
        tie_decision: rank_tie_decision(candidates, best, second, &score),
        midpoint_km: best_candidate.midpoint_km,
    }
}

fn rank_tie_decision(
    candidates: &[FaceCandidate],
    best: usize,
    second: Option<usize>,
    score: &impl Fn(&FaceCandidate) -> f64,
) -> R1RankTieDecision {
    let Some(second) = second else {
        return R1RankTieDecision::SoleEligibleFace;
    };
    let best = &candidates[best];
    let second = &candidates[second];
    if score(best).total_cmp(&score(second)) != Ordering::Equal {
        return R1RankTieDecision::Score;
    }
    if best.midpoint_km.x.total_cmp(&second.midpoint_km.x) != Ordering::Equal {
        return R1RankTieDecision::MidpointX;
    }
    if best.midpoint_km.y.total_cmp(&second.midpoint_km.y) != Ordering::Equal {
        return R1RankTieDecision::MidpointY;
    }
    if best.midpoint_km.z.total_cmp(&second.midpoint_km.z) != Ordering::Equal {
        return R1RankTieDecision::MidpointZ;
    }
    if best.portal_key != second.portal_key {
        return R1RankTieDecision::PortalKey;
    }
    R1RankTieDecision::CombinedFaceIndex
}

fn face_key_cmp(a: &FaceCandidate, b: &FaceCandidate) -> Ordering {
    a.midpoint_km
        .x
        .total_cmp(&b.midpoint_km.x)
        .then(a.midpoint_km.y.total_cmp(&b.midpoint_km.y))
        .then(a.midpoint_km.z.total_cmp(&b.midpoint_km.z))
        .then(a.portal_key.cmp(&b.portal_key))
        .then(a.combined_face_index.cmp(&b.combined_face_index))
}

#[derive(Debug, Clone, Copy)]
struct RouteAudit {
    maximum_fraction_sum_error: f64,
    maximum_cell_balance_error_km3_myr: f64,
}

fn audit_route(
    cap: &VoronoiCapFixture,
    flow: &FaceFlowCache,
    boundary_by_cell: &[Vec<usize>],
    reverse_edge: &[usize],
) -> Result<RouteAudit, R1CaseError> {
    let tolerance = 1.0e-12 * flow.total_supply_km3_myr;
    if !flow.total_supply_km3_myr.is_finite()
        || !flow.total_portal_outflow_km3_myr.is_finite()
        || !flow.total_sink_storage_km3_myr.is_finite()
        || flow.total_supply_km3_myr <= 0.0
        || flow.total_portal_outflow_km3_myr < 0.0
        || flow.total_sink_storage_km3_myr < 0.0
        || flow
            .portal_outflow_km3_myr
            .iter()
            .any(|(_, flux)| !flux.is_finite() || *flux < 0.0)
        || flow
            .directed_edge_fraction
            .iter()
            .chain(&flow.boundary_face_fraction)
            .any(|value| !value.is_finite() || *value < 0.0)
        || flow
            .directed_edge_flux_km3_myr
            .iter()
            .chain(&flow.boundary_face_flux_km3_myr)
            .chain(&flow.available_supply_km3_myr)
            .chain(&flow.sink_storage_km3_myr)
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(R1CaseError(
            "route contains a negative or non-finite water quantity".into(),
        ));
    }
    let mut maximum_fraction_sum_error = 0.0_f64;
    let mut maximum_cell_balance_error_km3_myr = 0.0_f64;
    for (cell, cell_boundary_faces) in boundary_by_cell
        .iter()
        .enumerate()
        .take(cap.mesh.cell_count())
    {
        let start = cap.mesh.edge_offsets[cell] as usize;
        let end = cap.mesh.edge_offsets[cell + 1] as usize;
        let internal_fraction: f64 = flow.directed_edge_fraction[start..end].iter().sum();
        let boundary_fraction: f64 = cell_boundary_faces
            .iter()
            .map(|&face| flow.boundary_face_fraction[face])
            .sum();
        let fraction_sum = internal_fraction + boundary_fraction;
        let donor = fraction_sum > 0.0;
        maximum_fraction_sum_error = maximum_fraction_sum_error.max(if donor {
            (fraction_sum - 1.0).abs()
        } else {
            fraction_sum.abs()
        });
        if maximum_fraction_sum_error > 1.0e-12 {
            return Err(R1CaseError(format!(
                "cell {cell} outgoing fractions do not normalize: {fraction_sum:e}"
            )));
        }

        let incoming: f64 = (start..end)
            .map(|edge| flow.directed_edge_flux_km3_myr[reverse_edge[edge]])
            .sum();
        let reconstructed_available = flow.local_supply_km3_myr[cell] + incoming;
        let cell_tolerance = 1.0e-12
            * flow.available_supply_km3_myr[cell]
                .abs()
                .max(reconstructed_available.abs())
                .max(1.0);
        if (flow.available_supply_km3_myr[cell] - reconstructed_available).abs() > cell_tolerance {
            return Err(R1CaseError(format!(
                "cell {cell} available supply does not equal local plus incoming water"
            )));
        }
        for edge in start..end {
            let expected_flux =
                flow.available_supply_km3_myr[cell] * flow.directed_edge_fraction[edge];
            if flow.directed_edge_flux_km3_myr[edge].to_bits() != expected_flux.to_bits() {
                return Err(R1CaseError(format!(
                    "cell {cell} edge {edge} flux is not available supply times fraction"
                )));
            }
        }
        for &face in cell_boundary_faces {
            let expected_flux =
                flow.available_supply_km3_myr[cell] * flow.boundary_face_fraction[face];
            if flow.boundary_face_flux_km3_myr[face].to_bits() != expected_flux.to_bits() {
                return Err(R1CaseError(format!(
                    "cell {cell} boundary face {face} flux is not available supply times fraction"
                )));
            }
        }
        let internal_out: f64 = flow.directed_edge_flux_km3_myr[start..end].iter().sum();
        let boundary_out: f64 = cell_boundary_faces
            .iter()
            .map(|&face| flow.boundary_face_flux_km3_myr[face])
            .sum();
        let residual = flow.local_supply_km3_myr[cell] + incoming
            - internal_out
            - boundary_out
            - flow.sink_storage_km3_myr[cell];
        maximum_cell_balance_error_km3_myr = maximum_cell_balance_error_km3_myr.max(residual.abs());
        if residual.abs() > cell_tolerance {
            return Err(R1CaseError(format!(
                "cell {cell} water balance residual {residual:e} exceeds {cell_tolerance:e}"
            )));
        }
        if donor {
            if flow.sink_storage_km3_myr[cell] != 0.0 {
                return Err(R1CaseError(format!(
                    "donor cell {cell} also stores sink water"
                )));
            }
        } else if flow.sink_storage_km3_myr[cell].to_bits()
            != flow.available_supply_km3_myr[cell].to_bits()
        {
            return Err(R1CaseError(format!(
                "sink cell {cell} does not store all available water"
            )));
        }
        for (edge, &reverse) in reverse_edge.iter().enumerate().take(end).skip(start) {
            if flow.directed_edge_flux_km3_myr[edge] > 0.0
                && flow.directed_edge_flux_km3_myr[reverse] > 0.0
            {
                return Err(R1CaseError(format!(
                    "reciprocal edge pair routes both directions at edge {edge}"
                )));
            }
        }
    }
    if flow.water_balance_error_km3_myr().abs() > tolerance {
        return Err(R1CaseError(format!(
            "global water balance residual {:e} exceeds {:e}",
            flow.water_balance_error_km3_myr(),
            tolerance
        )));
    }
    let mut portal_totals: Vec<_> = cap
        .mesh
        .outlet_portals
        .iter()
        .map(|portal| (portal.id, 0.0_f64))
        .collect();
    for (face_index, face) in cap.mesh.boundary_faces.iter().enumerate() {
        match face.condition {
            BoundaryFaceCondition::Closed => {
                if flow.boundary_face_fraction[face_index] != 0.0
                    || flow.boundary_face_flux_km3_myr[face_index] != 0.0
                {
                    return Err(R1CaseError(format!(
                        "closed boundary face {face_index} carries water"
                    )));
                }
            }
            BoundaryFaceCondition::OpenBaseLevel { portal_id, .. } => {
                let (_, total) = portal_totals
                    .iter_mut()
                    .find(|(id, _)| *id == portal_id)
                    .ok_or_else(|| R1CaseError(format!("unknown portal {portal_id:?}")))?;
                *total += flow.boundary_face_flux_km3_myr[face_index];
            }
        }
    }
    if portal_totals.len() != flow.portal_outflow_km3_myr.len() {
        return Err(R1CaseError("portal ledger length differs".into()));
    }
    for ((expected_id, expected_flux), (actual_id, actual_flux)) in
        portal_totals.iter().zip(&flow.portal_outflow_km3_myr)
    {
        if expected_id != actual_id || (expected_flux - actual_flux).abs() > tolerance {
            return Err(R1CaseError(format!(
                "portal ledger differs for {expected_id:?}"
            )));
        }
        if *actual_id != R1_CAP_PORTAL_ID && *actual_flux > 0.0 {
            return Err(R1CaseError(format!(
                "non-target portal {actual_id:?} receives water"
            )));
        }
    }
    let portal_total: f64 = portal_totals.iter().map(|(_, flux)| flux).sum();
    if (portal_total - flow.total_portal_outflow_km3_myr).abs() > tolerance {
        return Err(R1CaseError("portal totals do not close".into()));
    }
    let sink_total: f64 = flow.sink_storage_km3_myr.iter().sum();
    if (sink_total - flow.total_sink_storage_km3_myr).abs() > tolerance {
        return Err(R1CaseError("cell and total sink ledgers differ".into()));
    }
    Ok(RouteAudit {
        maximum_fraction_sum_error,
        maximum_cell_balance_error_km3_myr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landscape::{build_r1_voronoi_cap, VoronoiCapConfig};

    fn rectangle(x0: f64, x1: f64, y0: f64, y1: f64) -> Vec<DVec2> {
        vec![
            DVec2::new(x0, y0),
            DVec2::new(x1, y0),
            DVec2::new(x1, y1),
            DVec2::new(x0, y1),
        ]
    }

    #[test]
    fn polygon_moments_and_broad_clipping_are_exact_on_rectangles() {
        let polygon = rectangle(-2.0, 2.0, -3.0, 3.0);
        let moments = polygon_moments(&polygon).unwrap();
        assert_eq!(moments.area, 24.0);
        assert_eq!(moments.integral_x, 0.0);
        assert_eq!(moments.integral_y, 0.0);
        assert!((moments.integral_x2 / moments.area - 4.0 / 3.0).abs() < 1.0e-14);
        assert!((moments.integral_y2 / moments.area - 3.0).abs() < 1.0e-14);
        assert_eq!(moments.integral_xy, 0.0);
        let mut reversed = polygon.clone();
        reversed.reverse();
        let reversed_moments = polygon_moments(&reversed).unwrap();
        assert_eq!(moments.area, reversed_moments.area);
        assert_eq!(moments.integral_x2, reversed_moments.integral_x2);
        assert_eq!(moments.integral_y2, reversed_moments.integral_y2);

        let broad = rectangle(0.0, 2.0, -20.0, 20.0);
        let broad_mean =
            broad_exterior_integral(&broad).unwrap() / polygon_moments(&broad).unwrap().area;
        assert!((broad_mean - 128.0 / 15.0).abs() < 1.0e-12);
        assert_eq!(
            broad_exterior_integral(&rectangle(0.0, 2.0, -8.0, 8.0)).unwrap(),
            0.0
        );
        let upper = rectangle(0.0, 2.0, 14.0, 16.0);
        let upper_mean =
            broad_exterior_integral(&upper).unwrap() / polygon_moments(&upper).unwrap().area;
        assert!((upper_mean - 28.0 / 3.0).abs() < 1.0e-12);
        let lower = rectangle(0.0, 2.0, -16.0, -14.0);
        let lower_mean =
            broad_exterior_integral(&lower).unwrap() / polygon_moments(&lower).unwrap().area;
        assert!((lower_mean - 28.0 / 3.0).abs() < 1.0e-12);

        let translated_triangle = vec![
            DVec2::new(1.0, 2.0),
            DVec2::new(5.0, 2.0),
            DVec2::new(1.0, 8.0),
        ];
        let triangle = polygon_moments(&translated_triangle).unwrap();
        assert_eq!(triangle.area, 12.0);
        assert!((triangle.integral_x / triangle.area - 7.0 / 3.0).abs() < 1.0e-14);
        assert!((triangle.integral_y / triangle.area - 4.0).abs() < 1.0e-14);
        assert!((triangle.integral_x2 / triangle.area - 19.0 / 3.0).abs() < 1.0e-14);
        assert!((triangle.integral_y2 / triangle.area - 18.0).abs() < 1.0e-14);
        assert!((triangle.integral_xy / triangle.area - 26.0 / 3.0).abs() < 1.0e-14);

        let oblique_crossing = vec![
            DVec2::new(0.0, 10.0),
            DVec2::new(4.0, 10.0),
            DVec2::new(0.0, 16.0),
        ];
        assert!(
            (broad_exterior_integral(&oblique_crossing).unwrap() - 128.0 / 9.0).abs() < 1.0e-12
        );
    }

    #[test]
    fn convex_head_ownership_rejects_boundary_and_missing_points() {
        let square = rectangle(0.0, 1.0, 0.0, 1.0);
        validate_convex_polygon(&square, 1.0e-12).unwrap();
        assert_eq!(
            convex_point_relation(&square, DVec2::new(0.5, 0.5), 1.0e-12),
            PointRelation::Inside
        );
        assert_eq!(
            convex_point_relation(&square, DVec2::new(0.0, 0.5), 1.0e-12),
            PointRelation::Boundary
        );
        assert_eq!(
            convex_point_relation(&square, DVec2::new(2.0, 0.5), 1.0e-12),
            PointRelation::Outside
        );
        assert!(
            locate_unique_head(std::slice::from_ref(&square), DVec2::new(0.0, 0.5), 1.0e-12)
                .is_err()
        );
        assert!(locate_unique_head(&[square], DVec2::new(2.0, 0.5), 1.0e-12).is_err());
    }

    #[test]
    fn local_rank_records_the_decisive_tie_key() {
        let candidate = |edge, midpoint_x, grade| FaceCandidate {
            face: R1SelectedFace::Internal {
                directed_edge: edge,
                neighbor: edge + 10,
            },
            midpoint_km: DVec3::new(midpoint_x, 2.0, 0.0),
            portal_key: u32::MAX,
            combined_face_index: edge,
            physical_grade: grade,
            mfd_fraction: grade,
        };

        let sole = rank_candidates(&[candidate(1, 1.0, 2.0)], |face| face.physical_grade);
        assert_eq!(sole.tie_decision, R1RankTieDecision::SoleEligibleFace);

        let score = rank_candidates(&[candidate(1, 1.0, 2.0), candidate(2, 0.0, 1.0)], |face| {
            face.physical_grade
        });
        assert_eq!(score.tie_decision, R1RankTieDecision::Score);

        let midpoint = rank_candidates(&[candidate(1, 1.0, 2.0), candidate(2, 0.0, 2.0)], |face| {
            face.physical_grade
        });
        assert_eq!(midpoint.face, candidate(2, 0.0, 2.0).face);
        assert_eq!(midpoint.tie_decision, R1RankTieDecision::MidpointX);

        let build_index =
            rank_candidates(&[candidate(2, 1.0, 2.0), candidate(1, 1.0, 2.0)], |face| {
                face.physical_grade
            });
        assert_eq!(build_index.face, candidate(1, 1.0, 2.0).face);
        assert_eq!(
            build_index.tie_decision,
            R1RankTieDecision::CombinedFaceIndex
        );
    }

    #[test]
    fn affine_generator_point_control_is_explicit_deterministic_and_conservative() {
        let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        let config = R1CaseConfig {
            spacing_km: 8.0,
            theta_rad: 0.31,
            outlet_offset_km: 0.7,
            surface: R1SurfaceKind::Affine,
        };
        let first = build_r1_affine_generator_point_control(&cap, config).unwrap();
        let second = build_r1_affine_generator_point_control(&cap, config).unwrap();
        assert_eq!(first, second);

        let (outlet, along, _) = case_frame(config);
        for (cell, center) in cap.mesh.cell_center_km.iter().enumerate() {
            let point = DVec2::new(center.x, center.y);
            let expected =
                R1_BASE_ELEVATION_KM + R1_ALONG_TRACK_GRADE * (point - outlet).dot(along);
            assert_eq!(first.case.elevation_km[cell], expected);
        }
        assert_eq!(
            first.case.flow.routing_elevation_km,
            first.case.elevation_km
        );
        assert_eq!(
            first.case.flow.local_supply_km3_myr,
            first.case.local_supply_km3_myr
        );
        assert!(first.case.flow.flat_potential.iter().all(Option::is_none));
        assert!(first.case.audit.water_balance_relative_error <= 1.0e-12);
        assert!(first.case.audit.maximum_fraction_sum_error <= 1.0e-12);

        let registered = build_r1_registered_case(&cap, config).unwrap();
        assert!(first
            .case
            .elevation_km
            .iter()
            .zip(&registered.elevation_km)
            .any(|(generator, polygon_mean)| generator.to_bits() != polygon_mean.to_bits()));

        let error = build_r1_affine_generator_point_control(
            &cap,
            R1CaseConfig {
                surface: R1SurfaceKind::Valley,
                ..config
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("only for the affine surface"));
    }

    #[test]
    fn registered_eight_km_inputs_route_and_expose_domain_rank_conflicts() {
        let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        let cap_before = cap.clone();
        let mut av_conflicts = 0usize;
        let mut av_head_conflicts = 0usize;
        for surface in [
            R1SurfaceKind::Affine,
            R1SurfaceKind::Valley,
            R1SurfaceKind::Broad,
        ] {
            for theta_rad in [0.0, 0.31] {
                for outlet_offset_km in [0.0, 0.7] {
                    let case = build_r1_registered_case(
                        &cap,
                        R1CaseConfig {
                            spacing_km: 8.0,
                            theta_rad,
                            outlet_offset_km,
                            surface,
                        },
                    )
                    .unwrap();
                    assert!(case.audit.water_balance_relative_error <= 1.0e-12);
                    assert!(case.audit.maximum_fraction_sum_error <= 1.0e-12);
                    assert_eq!(
                        case.heads.len(),
                        usize::from(surface == R1SurfaceKind::Broad) + 1
                    );
                    if surface != R1SurfaceKind::Broad {
                        av_conflicts += case.audit.domain_rank_conflicts;
                        av_head_conflicts += case.audit.head_rank_conflicts;
                    }
                }
            }
        }
        assert!(av_conflicts > 0);
        assert!(av_head_conflicts > 0);
        assert_eq!(cap, cap_before);
    }

    #[test]
    #[ignore = "full registered 8/4/2 km input and rank matrix is an audit test"]
    fn registered_full_input_matrix_passes_and_reports_rank_precheck() {
        let mut total_av_conflicts = 0usize;
        let mut total_av_head_conflicts = 0usize;
        for spacing_km in [8.0, 4.0, 2.0] {
            let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(spacing_km)).unwrap();
            let cap_before = cap.clone();
            for surface in [
                R1SurfaceKind::Affine,
                R1SurfaceKind::Valley,
                R1SurfaceKind::Broad,
            ] {
                for theta_rad in [0.0, 0.31] {
                    for outlet_offset_km in [0.0, 0.7] {
                        let case = build_r1_registered_case(
                            &cap,
                            R1CaseConfig {
                                spacing_km,
                                theta_rad,
                                outlet_offset_km,
                                surface,
                            },
                        )
                        .unwrap();
                        let repeated = build_r1_registered_case(&cap, case.config).unwrap();
                        assert_eq!(case, repeated);
                        if surface != R1SurfaceKind::Broad {
                            total_av_conflicts += case.audit.domain_rank_conflicts;
                            total_av_head_conflicts += case.audit.head_rank_conflicts;
                        }
                        eprintln!(
                            "spacing={spacing_km} theta={theta_rad} delta={outlet_offset_km} surface={surface:?} heads={:?} donors={} conflicts={} head_conflicts={} p0_margin_min={:.6e} m0_margin_min={:.6e} fraction_error_max={:.6e} cell_balance_max={:.6e} water_rel={:.6e} portal={:.6e} sink={:.6e}",
                            case.heads.iter().map(|head| head.cell).collect::<Vec<_>>(),
                            case.audit.donor_cells,
                            case.audit.domain_rank_conflicts,
                            case.audit.head_rank_conflicts,
                            case.audit.minimum_p0_normalized_margin,
                            case.audit.minimum_m0_normalized_margin,
                            case.audit.maximum_fraction_sum_error,
                            case.audit.maximum_cell_balance_error_km3_myr,
                            case.audit.water_balance_relative_error,
                            case.audit.target_portal_outflow_km3_myr,
                            case.audit.total_sink_storage_km3_myr,
                        );
                    }
                }
            }
            assert_eq!(cap, cap_before);
        }
        assert!(total_av_conflicts > 0);
        assert!(total_av_head_conflicts > 0);
    }

    #[test]
    fn registered_case_is_bit_deterministic() {
        let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        let config = R1CaseConfig {
            spacing_km: 8.0,
            theta_rad: 0.31,
            outlet_offset_km: 0.7,
            surface: R1SurfaceKind::Valley,
        };
        let first = build_r1_registered_case(&cap, config).unwrap();
        let second = build_r1_registered_case(&cap, config).unwrap();
        assert_eq!(first, second);

        let mut comparison_only_cap = cap.clone();
        comparison_only_cap.config.guard_spacings = 10;
        assert!(build_r1_registered_case(&comparison_only_cap, config).is_err());
    }
}
