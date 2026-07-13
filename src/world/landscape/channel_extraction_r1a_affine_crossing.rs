//! Test-only continuous polygon crossing for the registered affine R1a fixture.
//!
//! X0 is the analytic direction oracle. X1 reconstructs an affine gradient
//! from exact polygon means at polygon centroids. Neither arm is a receiver,
//! water-routing, or product implementation.

use std::cmp::Ordering;

use glam::DVec2;

use super::{
    build_r1_registered_case, build_r1_voronoi_cap,
    channel_extraction_r1a_fixture::build_r1_affine_generator_point_control, BoundaryFaceCondition,
    OutletPortalId, R1CaseConfig, R1CaseError, R1RegisteredCase, R1SurfaceKind, VoronoiCapConfig,
    VoronoiCapFixture, R1_CAP_HEIGHT_KM, R1_CAP_PORTAL_ID, R1_HEAD_ALONG_TRACK_KM,
};

const RELATIVE_TOLERANCE: f64 = 1.0e-10;
const RECONSTRUCTION_SINGULAR_FACTOR: f64 = 1.0e-12;
const ANALYTIC_GRADE: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossingArm {
    X0Analytic,
    X1CentroidReconstruction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentFace {
    Internal {
        directed_edge: usize,
        neighbor: usize,
        reciprocal_edge: usize,
    },
    Boundary {
        boundary_face: usize,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct CheckedSegment {
    a_km: DVec2,
    b_km: DVec2,
    face: SegmentFace,
}

#[derive(Debug, Clone, PartialEq)]
struct CheckedCell {
    signed_area_km2: f64,
    centroid_km: DVec2,
    segments: Vec<CheckedSegment>,
}

#[derive(Debug, Clone, PartialEq)]
struct CheckedSegmentContext {
    spacing_km: f64,
    cells: Vec<CheckedCell>,
    edge_segment: Vec<(usize, usize)>,
    boundary_segment: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq)]
struct ReconstructionAudit {
    gradients: Vec<DVec2>,
    relative_vector_errors: Vec<f64>,
    sine_angle_errors: Vec<f64>,
    maximum_relative_vector_error: f64,
    maximum_sine_angle_error: f64,
}

impl ReconstructionAudit {
    fn passes_gate(&self) -> bool {
        self.maximum_relative_vector_error <= RELATIVE_TOLERANCE
            && self.maximum_sine_angle_error <= RELATIVE_TOLERANCE
    }
}

#[derive(Debug, Clone, PartialEq)]
struct InternalScoreCellAudit {
    generator_winner: Option<usize>,
    reconstructed_winner: Option<usize>,
    generator_normalized_margin: Option<f64>,
    reconstructed_normalized_margin: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct InternalScoreEquivalenceAudit {
    cells: Vec<InternalScoreCellAudit>,
    eligibility_conflicts: usize,
    winner_conflicts: usize,
    minimum_generator_normalized_margin: Option<f64>,
    minimum_reconstructed_normalized_margin: Option<f64>,
    minimum_absolute_generator_score: f64,
    maximum_symmetric_normalized_score_error: f64,
}

impl InternalScoreEquivalenceAudit {
    fn passes_gate(&self) -> bool {
        self.eligibility_conflicts == 0 && self.winner_conflicts == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossingTermination {
    ReachedPortal,
    ClosedBoundary {
        cell: usize,
        boundary_face: usize,
    },
    WrongPortal {
        cell: usize,
        boundary_face: usize,
        portal_id: OutletPortalId,
    },
    MissingExit {
        cell: usize,
    },
    NonAdvancing {
        cell: usize,
        entry_segment: usize,
    },
    TangentEntryAmbiguity {
        cell: usize,
        entry_segment: usize,
    },
    CollinearAmbiguity {
        cell: usize,
        segment: usize,
    },
    VertexAmbiguity {
        cell: usize,
    },
    RepeatedCell {
        cell: usize,
    },
    CellCountGuard,
}

#[derive(Debug, Clone, PartialEq)]
struct CrossingRecord {
    cell: usize,
    segment: usize,
    face: SegmentFace,
    point_km: DVec2,
    ray_parameter_km: f64,
    segment_parameter: f64,
    vertex_clearance_km: f64,
    first_second_exit_parameter_gap_km: Option<f64>,
    segment_residual_km: f64,
    reciprocal_segment_residual_km: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct CrossingMetrics {
    maximum_absolute_cross_track_km: f64,
    polyline_arclength_km: f64,
    relative_arclength_error: f64,
    total_backtracking_km: f64,
    endpoint_error_km: f64,
    terminal_along_track_km: f64,
    terminal_cross_track_km: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct CrossingMetricDifferences {
    maximum_absolute_cross_track_km: f64,
    polyline_arclength_km: f64,
    relative_arclength_error: f64,
    total_backtracking_km: f64,
    endpoint_error_km: f64,
    terminal_along_track_km: f64,
    terminal_cross_track_km: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct CrossingTrace {
    arm: CrossingArm,
    termination: CrossingTermination,
    visited_cells: Vec<usize>,
    crossings: Vec<CrossingRecord>,
    vertices_km: Vec<DVec2>,
    maximum_segment_residual_km: f64,
    minimum_vertex_clearance_km: Option<f64>,
    minimum_exit_parameter_gap_km: Option<f64>,
    metrics: Option<CrossingMetrics>,
    visited_maximum_relative_gradient_error: Option<f64>,
    visited_maximum_sine_angle_error: Option<f64>,
    all_domain_maximum_relative_gradient_error: Option<f64>,
    all_domain_maximum_sine_angle_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct PairedCrossingObservation {
    x0: CrossingTrace,
    x1: Option<CrossingTrace>,
    face_sequences_equal: Option<bool>,
    endpoint_difference_km: Option<f64>,
    per_crossing_maximum_difference_km: Option<f64>,
    metric_differences: Option<CrossingMetricDifferences>,
}

impl CheckedSegmentContext {
    fn new(cap: &VoronoiCapFixture) -> Result<Self, R1CaseError> {
        cap.mesh
            .validate()
            .map_err(|error| R1CaseError(error.to_string()))?;
        let cell_count = cap.mesh.cell_count();
        if cap.cell_polygons_km.len() != cell_count
            || cap.edge_face_midpoint_km.len() != cap.mesh.edge_neighbor.len()
        {
            return Err(R1CaseError(
                "affine crossing cap arrays have incompatible lengths".into(),
            ));
        }

        let mut boundary_by_cell = vec![Vec::new(); cell_count];
        for (face_index, face) in cap.mesh.boundary_faces.iter().enumerate() {
            let owner = face.cell as usize;
            let Some(faces) = boundary_by_cell.get_mut(owner) else {
                return Err(R1CaseError(format!(
                    "boundary face {face_index} has invalid owner {owner}"
                )));
            };
            faces.push(face_index);
        }
        let reciprocal_edges = reciprocal_edges(cap)?;
        let mut edge_consumed = vec![false; cap.mesh.edge_neighbor.len()];
        let mut boundary_consumed = vec![false; cap.mesh.boundary_faces.len()];
        let mut edge_segment = vec![(usize::MAX, usize::MAX); cap.mesh.edge_neighbor.len()];
        let mut boundary_segment = vec![(usize::MAX, usize::MAX); cap.mesh.boundary_faces.len()];
        let mut cells = Vec::with_capacity(cell_count);

        for (cell, polygon) in cap.cell_polygons_km.iter().enumerate() {
            let (signed_area, centroid) = signed_area_and_centroid(polygon)?;
            if (signed_area.abs() - cap.mesh.cell_area_km2[cell]).abs()
                > RELATIVE_TOLERANCE * cap.config.spacing_km.powi(2)
            {
                return Err(R1CaseError(format!(
                    "cell {cell} polygon area disagrees with the mesh"
                )));
            }
            let edge_start = cap.mesh.edge_offsets[cell] as usize;
            let edge_end = cap.mesh.edge_offsets[cell + 1] as usize;
            if polygon.len() != edge_end - edge_start + boundary_by_cell[cell].len() {
                return Err(R1CaseError(format!(
                    "cell {cell} polygon segment and owned face counts differ"
                )));
            }
            let mut segments = Vec::with_capacity(polygon.len());
            for (segment_index, (&a, &b)) in polygon
                .iter()
                .zip(polygon.iter().cycle().skip(1))
                .take(polygon.len())
                .enumerate()
            {
                let midpoint = 0.5 * (a + b);
                let internal_matches: Vec<_> = (edge_start..edge_end)
                    .filter(|&edge| {
                        let stored = cap.edge_face_midpoint_km[edge];
                        stored == midpoint.extend(0.0)
                    })
                    .collect();
                let boundary_matches: Vec<_> = boundary_by_cell[cell]
                    .iter()
                    .copied()
                    .filter(|&face| cap.mesh.boundary_faces[face].center_km == midpoint.extend(0.0))
                    .collect();
                if internal_matches.len() + boundary_matches.len() != 1 {
                    return Err(R1CaseError(format!(
                        "cell {cell} segment {segment_index} has {} exact face matches",
                        internal_matches.len() + boundary_matches.len()
                    )));
                }
                let face = if let Some(&edge) = internal_matches.first() {
                    if std::mem::replace(&mut edge_consumed[edge], true) {
                        return Err(R1CaseError(format!(
                            "directed edge {edge} was consumed twice"
                        )));
                    }
                    edge_segment[edge] = (cell, segment_index);
                    SegmentFace::Internal {
                        directed_edge: edge,
                        neighbor: cap.mesh.edge_neighbor[edge] as usize,
                        reciprocal_edge: reciprocal_edges[edge],
                    }
                } else {
                    let face = boundary_matches[0];
                    if std::mem::replace(&mut boundary_consumed[face], true) {
                        return Err(R1CaseError(format!(
                            "boundary face {face} was consumed twice"
                        )));
                    }
                    boundary_segment[face] = (cell, segment_index);
                    SegmentFace::Boundary {
                        boundary_face: face,
                    }
                };
                segments.push(CheckedSegment {
                    a_km: a,
                    b_km: b,
                    face,
                });
            }
            cells.push(CheckedCell {
                signed_area_km2: signed_area,
                centroid_km: centroid,
                segments,
            });
        }
        if edge_consumed.iter().any(|consumed| !consumed)
            || boundary_consumed.iter().any(|consumed| !consumed)
        {
            return Err(R1CaseError(
                "checked segment context did not consume every stored face".into(),
            ));
        }
        for edge in 0..cap.mesh.edge_neighbor.len() {
            let (cell, segment) = edge_segment[edge];
            let reverse = reciprocal_edges[edge];
            let (neighbor, reverse_segment) = edge_segment[reverse];
            if neighbor != cap.mesh.edge_neighbor[edge] as usize {
                return Err(R1CaseError(format!(
                    "edge {edge} reciprocal segment has the wrong owner"
                )));
            }
            let here = &cells[cell].segments[segment];
            let there = &cells[neighbor].segments[reverse_segment];
            if !same_segment_endpoints(here, there) {
                return Err(R1CaseError(format!(
                    "edge {edge} reciprocal polygon endpoints disagree"
                )));
            }
        }
        Ok(Self {
            spacing_km: cap.config.spacing_km,
            cells,
            edge_segment,
            boundary_segment,
        })
    }
}

fn reciprocal_edges(cap: &VoronoiCapFixture) -> Result<Vec<usize>, R1CaseError> {
    let mut reverse = vec![usize::MAX; cap.mesh.edge_neighbor.len()];
    for cell in 0..cap.mesh.cell_count() {
        let start = cap.mesh.edge_offsets[cell] as usize;
        let end = cap.mesh.edge_offsets[cell + 1] as usize;
        for (edge, reverse_edge) in reverse.iter_mut().enumerate().take(end).skip(start) {
            let neighbor = cap.mesh.edge_neighbor[edge] as usize;
            let matches: Vec<_> = (cap.mesh.edge_offsets[neighbor] as usize
                ..cap.mesh.edge_offsets[neighbor + 1] as usize)
                .filter(|&candidate| cap.mesh.edge_neighbor[candidate] as usize == cell)
                .collect();
            if matches.len() != 1 {
                return Err(R1CaseError(format!(
                    "edge {edge} has {} reciprocal CSR matches",
                    matches.len()
                )));
            }
            *reverse_edge = matches[0];
        }
    }
    Ok(reverse)
}

fn signed_area_and_centroid(polygon: &[DVec2]) -> Result<(f64, DVec2), R1CaseError> {
    if polygon.len() < 3 {
        return Err(R1CaseError(
            "crossing polygon has fewer than three vertices".into(),
        ));
    }
    let mut twice_area = 0.0;
    let mut centroid_numerator = DVec2::ZERO;
    for (&a, &b) in polygon
        .iter()
        .zip(polygon.iter().cycle().skip(1))
        .take(polygon.len())
    {
        let cross = a.perp_dot(b);
        twice_area += cross;
        centroid_numerator += (a + b) * cross;
    }
    let signed_area = 0.5 * twice_area;
    if !signed_area.is_finite() || signed_area == 0.0 {
        return Err(R1CaseError(
            "crossing polygon has invalid signed area".into(),
        ));
    }
    let centroid = centroid_numerator / (3.0 * twice_area);
    if !centroid.is_finite() {
        return Err(R1CaseError(
            "crossing polygon centroid is non-finite".into(),
        ));
    }
    Ok((signed_area, centroid))
}

fn same_segment_endpoints(a: &CheckedSegment, b: &CheckedSegment) -> bool {
    (a.a_km == b.a_km && a.b_km == b.b_km) || (a.a_km == b.b_km && a.b_km == b.a_km)
}

fn reconstruct_centroid_gradients(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    context: &CheckedSegmentContext,
) -> Result<ReconstructionAudit, R1CaseError> {
    validate_affine_case(cap, case, context)?;
    let (_, along, _) = case_frame(case.config);
    let analytic = ANALYTIC_GRADE * along;
    let h = case.config.spacing_km;
    let mut gradients = Vec::with_capacity(cap.mesh.cell_count());
    let mut relative_vector_errors = Vec::with_capacity(cap.mesh.cell_count());
    let mut sine_angle_errors = Vec::with_capacity(cap.mesh.cell_count());
    let mut maximum_relative_vector_error = 0.0_f64;
    let mut maximum_sine_angle_error = 0.0_f64;
    for cell in 0..cap.mesh.cell_count() {
        let mut m_xx = 0.0;
        let mut m_xy = 0.0;
        let mut m_yy = 0.0;
        let mut b_x = 0.0;
        let mut b_y = 0.0;
        let start = cap.mesh.edge_offsets[cell] as usize;
        let end = cap.mesh.edge_offsets[cell + 1] as usize;
        for edge in start..end {
            let neighbor = cap.mesh.edge_neighbor[edge] as usize;
            let r = (context.cells[neighbor].centroid_km - context.cells[cell].centroid_km) / h;
            let q = (case.elevation_km[neighbor] - case.elevation_km[cell]) / h;
            m_xx += r.x * r.x;
            m_xy += r.x * r.y;
            m_yy += r.y * r.y;
            b_x += r.x * q;
            b_y += r.y * q;
        }
        let determinant = m_xx * m_yy - m_xy * m_xy;
        let trace = m_xx + m_yy;
        if !determinant.is_finite()
            || !trace.is_finite()
            || !b_x.is_finite()
            || !b_y.is_finite()
            || determinant <= RECONSTRUCTION_SINGULAR_FACTOR * trace * trace
        {
            return Err(R1CaseError(format!(
                "centroid reconstruction is singular at cell {cell}"
            )));
        }
        let gradient = DVec2::new(
            (m_yy * b_x - m_xy * b_y) / determinant,
            (m_xx * b_y - m_xy * b_x) / determinant,
        );
        if !gradient.is_finite() || gradient.dot(analytic) <= 0.0 {
            return Err(R1CaseError(format!(
                "centroid reconstruction is non-finite or misaligned at cell {cell}"
            )));
        }
        let relative_error = gradient.distance(analytic) / analytic.length();
        let sine_error =
            gradient.perp_dot(analytic).abs() / (gradient.length() * analytic.length());
        maximum_relative_vector_error = maximum_relative_vector_error.max(relative_error);
        maximum_sine_angle_error = maximum_sine_angle_error.max(sine_error);
        gradients.push(gradient);
        relative_vector_errors.push(relative_error);
        sine_angle_errors.push(sine_error);
    }
    Ok(ReconstructionAudit {
        gradients,
        relative_vector_errors,
        sine_angle_errors,
        maximum_relative_vector_error,
        maximum_sine_angle_error,
    })
}

fn audit_internal_score_equivalence(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    reconstruction: &ReconstructionAudit,
) -> Result<InternalScoreEquivalenceAudit, R1CaseError> {
    if case.config.surface != R1SurfaceKind::Affine {
        return Err(R1CaseError("score equivalence requires affine A".into()));
    }
    let generator = build_r1_affine_generator_point_control(cap, case.config)?.case;
    let mut cells = Vec::with_capacity(cap.mesh.cell_count());
    let mut eligibility_conflicts = 0usize;
    let mut winner_conflicts = 0usize;
    let mut minimum_generator_normalized_margin: Option<f64> = None;
    let mut minimum_reconstructed_normalized_margin: Option<f64> = None;
    let mut minimum_absolute_generator_score = f64::INFINITY;
    let mut maximum_symmetric_normalized_score_error = 0.0_f64;
    for cell in 0..cap.mesh.cell_count() {
        let start = cap.mesh.edge_offsets[cell] as usize;
        let end = cap.mesh.edge_offsets[cell + 1] as usize;
        let mut generator_scores = Vec::with_capacity(end - start);
        let mut reconstructed_scores = Vec::with_capacity(end - start);
        for edge in start..end {
            let neighbor = cap.mesh.edge_neighbor[edge] as usize;
            let distance = f64::from(cap.mesh.edge_distance_km[edge]);
            let generator_score =
                (generator.elevation_km[cell] - generator.elevation_km[neighbor]) / distance;
            let connector = DVec2::new(
                cap.mesh.cell_center_km[neighbor].x - cap.mesh.cell_center_km[cell].x,
                cap.mesh.cell_center_km[neighbor].y - cap.mesh.cell_center_km[cell].y,
            );
            let reconstructed_score = -reconstruction.gradients[cell].dot(connector) / distance;
            if !generator_score.is_finite() || !reconstructed_score.is_finite() {
                return Err(R1CaseError(format!(
                    "non-finite internal score at edge {edge}"
                )));
            }
            eligibility_conflicts +=
                usize::from((generator_score > 0.0) != (reconstructed_score > 0.0));
            minimum_absolute_generator_score =
                minimum_absolute_generator_score.min(generator_score.abs());
            maximum_symmetric_normalized_score_error = maximum_symmetric_normalized_score_error
                .max(
                    (reconstructed_score - generator_score).abs()
                        / reconstructed_score
                            .abs()
                            .max(generator_score.abs())
                            .max(f64::MIN_POSITIVE),
                );
            generator_scores.push((edge, generator_score));
            reconstructed_scores.push((edge, reconstructed_score));
        }
        let (generator_winner, generator_margin) = rank_internal(cap, &generator_scores);
        let (reconstructed_winner, reconstructed_margin) =
            rank_internal(cap, &reconstructed_scores);
        winner_conflicts += usize::from(generator_winner != reconstructed_winner);
        if let Some(margin) = generator_margin {
            minimum_generator_normalized_margin = Some(
                minimum_generator_normalized_margin.map_or(margin, |current| current.min(margin)),
            );
        }
        if let Some(margin) = reconstructed_margin {
            minimum_reconstructed_normalized_margin = Some(
                minimum_reconstructed_normalized_margin
                    .map_or(margin, |current| current.min(margin)),
            );
        }
        cells.push(InternalScoreCellAudit {
            generator_winner,
            reconstructed_winner,
            generator_normalized_margin: generator_margin,
            reconstructed_normalized_margin: reconstructed_margin,
        });
    }
    Ok(InternalScoreEquivalenceAudit {
        cells,
        eligibility_conflicts,
        winner_conflicts,
        minimum_generator_normalized_margin,
        minimum_reconstructed_normalized_margin,
        minimum_absolute_generator_score,
        maximum_symmetric_normalized_score_error,
    })
}

fn rank_internal(cap: &VoronoiCapFixture, scores: &[(usize, f64)]) -> (Option<usize>, Option<f64>) {
    let mut eligible: Vec<_> = scores
        .iter()
        .copied()
        .filter(|(_, score)| *score > 0.0)
        .collect();
    eligible.sort_by(|(edge_a, score_a), (edge_b, score_b)| {
        score_b
            .total_cmp(score_a)
            .then_with(|| midpoint_edge_cmp(cap, *edge_a, *edge_b))
    });
    let Some(&(winner, best)) = eligible.first() else {
        return (None, None);
    };
    let second = eligible.get(1).map_or(0.0, |(_, score)| *score);
    (
        Some(winner),
        Some((best - second) / best.abs().max(f64::MIN_POSITIVE)),
    )
}

fn midpoint_edge_cmp(cap: &VoronoiCapFixture, a: usize, b: usize) -> Ordering {
    let a_midpoint = cap.edge_face_midpoint_km[a];
    let b_midpoint = cap.edge_face_midpoint_km[b];
    a_midpoint
        .x
        .total_cmp(&b_midpoint.x)
        .then(a_midpoint.y.total_cmp(&b_midpoint.y))
        .then(a_midpoint.z.total_cmp(&b_midpoint.z))
        .then(a.cmp(&b))
}

fn trace_crossing(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    context: &CheckedSegmentContext,
    reconstruction: &ReconstructionAudit,
    arm: CrossingArm,
    required_portal: OutletPortalId,
) -> Result<CrossingTrace, R1CaseError> {
    validate_affine_case(cap, case, context)?;
    if reconstruction.gradients.len() != cap.mesh.cell_count() {
        return Err(R1CaseError(
            "gradient context has incompatible length".into(),
        ));
    }
    if arm == CrossingArm::X1CentroidReconstruction && !reconstruction.passes_gate() {
        return Err(R1CaseError("X1 reconstruction gate did not pass".into()));
    }
    if !cap
        .mesh
        .outlet_portals
        .iter()
        .any(|portal| portal.id == required_portal)
    {
        return Err(R1CaseError("required crossing portal is absent".into()));
    }
    let head = case
        .heads
        .first()
        .ok_or_else(|| R1CaseError("affine crossing case has no head".into()))?;
    let (_, along, _) = case_frame(case.config);
    let tau_x = RELATIVE_TOLERANCE * context.spacing_km;
    let mut cell = head.cell;
    let mut entry_segment: Option<usize> = None;
    let mut visited = vec![false; cap.mesh.cell_count()];
    let mut visited_cells = Vec::new();
    let mut crossings = Vec::new();
    let mut vertices = vec![head.point_km];
    let mut point = head.point_km;
    let termination;

    loop {
        if visited_cells.len() >= cap.mesh.cell_count() {
            termination = CrossingTermination::CellCountGuard;
            break;
        }
        if visited[cell] {
            termination = CrossingTermination::RepeatedCell { cell };
            break;
        }
        visited[cell] = true;
        visited_cells.push(cell);
        let direction = match arm {
            CrossingArm::X0Analytic => -along,
            CrossingArm::X1CentroidReconstruction => {
                -reconstruction.gradients[cell] / reconstruction.gradients[cell].length()
            }
        };
        if let Some(segment_index) = entry_segment {
            let entry = &context.cells[cell].segments[segment_index];
            let edge = entry.b_km - entry.a_km;
            let inward = context.cells[cell].signed_area_km2.signum() * edge.perp_dot(direction);
            let threshold = RELATIVE_TOLERANCE * edge.length();
            if inward.abs() <= threshold {
                termination = CrossingTermination::TangentEntryAmbiguity {
                    cell,
                    entry_segment: segment_index,
                };
                break;
            }
            if inward < 0.0 {
                termination = CrossingTermination::NonAdvancing {
                    cell,
                    entry_segment: segment_index,
                };
                break;
            }
        }

        let mut eligible = Vec::new();
        let mut collinear = None;
        for (segment_index, segment) in context.cells[cell].segments.iter().enumerate() {
            if Some(segment_index) == entry_segment {
                continue;
            }
            let edge = segment.b_km - segment.a_km;
            let edge_length = edge.length();
            let denominator = direction.perp_dot(edge);
            if denominator.abs() <= RELATIVE_TOLERANCE * edge_length {
                if (segment.a_km - point).perp_dot(edge).abs() / edge_length <= tau_x {
                    collinear = Some(segment_index);
                    break;
                }
                continue;
            }
            let relative = segment.a_km - point;
            let t = relative.perp_dot(edge) / denominator;
            let u = relative.perp_dot(direction) / denominator;
            if t > tau_x && (-RELATIVE_TOLERANCE..=1.0 + RELATIVE_TOLERANCE).contains(&u) {
                let intersection = point + t * direction;
                let clearance = intersection
                    .distance(segment.a_km)
                    .min(intersection.distance(segment.b_km));
                eligible.push((t, segment_index, u, intersection, clearance));
            }
        }
        if let Some(segment) = collinear {
            termination = CrossingTermination::CollinearAmbiguity { cell, segment };
            break;
        }
        if eligible.is_empty() {
            termination = CrossingTermination::MissingExit { cell };
            break;
        }
        eligible.sort_by(|a, b| a.0.total_cmp(&b.0));
        if eligible.iter().any(|candidate| candidate.4 <= tau_x)
            || eligible
                .windows(2)
                .any(|pair| (pair[1].0 - pair[0].0).abs() <= tau_x)
        {
            termination = CrossingTermination::VertexAmbiguity { cell };
            break;
        }
        let (t, segment_index, u, intersection, clearance) = eligible[0];
        let exit_gap = eligible.get(1).map(|candidate| candidate.0 - t);
        let segment = &context.cells[cell].segments[segment_index];
        let residual = point_segment_line_residual(intersection, segment);
        if residual > tau_x {
            return Err(R1CaseError(format!(
                "cell {cell} crossing misses its declared segment by {residual:e} km"
            )));
        }
        let mut reciprocal_residual = None;
        if let SegmentFace::Internal {
            reciprocal_edge, ..
        } = segment.face
        {
            let (neighbor, reciprocal_segment) = context.edge_segment[reciprocal_edge];
            let reciprocal = &context.cells[neighbor].segments[reciprocal_segment];
            let reciprocal_line_residual = point_segment_line_residual(intersection, reciprocal);
            let reciprocal_parameter = point_segment_parameter(intersection, reciprocal);
            if reciprocal_line_residual > tau_x
                || !(-RELATIVE_TOLERANCE..=1.0 + RELATIVE_TOLERANCE).contains(&reciprocal_parameter)
            {
                return Err(R1CaseError(format!(
                    "cell {cell} crossing is not on reciprocal segment {reciprocal_segment}"
                )));
            }
            reciprocal_residual = Some(reciprocal_line_residual);
        }
        crossings.push(CrossingRecord {
            cell,
            segment: segment_index,
            face: segment.face,
            point_km: intersection,
            ray_parameter_km: t,
            segment_parameter: u,
            vertex_clearance_km: clearance,
            first_second_exit_parameter_gap_km: exit_gap,
            segment_residual_km: residual,
            reciprocal_segment_residual_km: reciprocal_residual,
        });
        vertices.push(intersection);
        point = intersection;

        match segment.face {
            SegmentFace::Internal {
                neighbor,
                reciprocal_edge,
                ..
            } => {
                let (owner, reciprocal_segment) = context.edge_segment[reciprocal_edge];
                if owner != neighbor {
                    return Err(R1CaseError(
                        "internal crossing reciprocal owner changed".into(),
                    ));
                }
                if visited[neighbor] {
                    termination = CrossingTermination::RepeatedCell { cell: neighbor };
                    break;
                }
                cell = neighbor;
                entry_segment = Some(reciprocal_segment);
            }
            SegmentFace::Boundary { boundary_face } => {
                match cap.mesh.boundary_faces[boundary_face].condition {
                    BoundaryFaceCondition::Closed => {
                        termination = CrossingTermination::ClosedBoundary {
                            cell,
                            boundary_face,
                        };
                    }
                    BoundaryFaceCondition::OpenBaseLevel { portal_id, .. }
                        if portal_id == required_portal =>
                    {
                        termination = CrossingTermination::ReachedPortal;
                    }
                    BoundaryFaceCondition::OpenBaseLevel { portal_id, .. } => {
                        termination = CrossingTermination::WrongPortal {
                            cell,
                            boundary_face,
                            portal_id,
                        };
                    }
                }
                break;
            }
        }
    }

    let maximum_segment_residual_km = crossings.iter().fold(0.0_f64, |maximum, crossing| {
        maximum
            .max(crossing.segment_residual_km)
            .max(crossing.reciprocal_segment_residual_km.unwrap_or(0.0))
    });
    let minimum_vertex_clearance_km = crossings
        .iter()
        .map(|crossing| crossing.vertex_clearance_km)
        .reduce(f64::min);
    let minimum_exit_parameter_gap_km = crossings
        .iter()
        .filter_map(|crossing| crossing.first_second_exit_parameter_gap_km)
        .reduce(f64::min);
    let metrics = (termination == CrossingTermination::ReachedPortal)
        .then(|| crossing_metrics(&vertices, case.config));
    let (visited_maximum_relative_gradient_error, visited_maximum_sine_angle_error) =
        if arm == CrossingArm::X1CentroidReconstruction {
            (
                visited_cells
                    .iter()
                    .map(|&visited| reconstruction.relative_vector_errors[visited])
                    .reduce(f64::max),
                visited_cells
                    .iter()
                    .map(|&visited| reconstruction.sine_angle_errors[visited])
                    .reduce(f64::max),
            )
        } else {
            (None, None)
        };
    Ok(CrossingTrace {
        arm,
        termination,
        visited_cells,
        crossings,
        vertices_km: vertices,
        maximum_segment_residual_km,
        minimum_vertex_clearance_km,
        minimum_exit_parameter_gap_km,
        metrics,
        visited_maximum_relative_gradient_error,
        visited_maximum_sine_angle_error,
        all_domain_maximum_relative_gradient_error: (arm == CrossingArm::X1CentroidReconstruction)
            .then_some(reconstruction.maximum_relative_vector_error),
        all_domain_maximum_sine_angle_error: (arm == CrossingArm::X1CentroidReconstruction)
            .then_some(reconstruction.maximum_sine_angle_error),
    })
}

fn point_segment_line_residual(point: DVec2, segment: &CheckedSegment) -> f64 {
    let edge = segment.b_km - segment.a_km;
    (point - segment.a_km).perp_dot(edge).abs() / edge.length()
}

fn point_segment_parameter(point: DVec2, segment: &CheckedSegment) -> f64 {
    let edge = segment.b_km - segment.a_km;
    (point - segment.a_km).dot(edge) / edge.length_squared()
}

fn crossing_metrics(vertices: &[DVec2], config: R1CaseConfig) -> CrossingMetrics {
    let (outlet, along, transverse) = case_frame(config);
    let coordinates: Vec<_> = vertices
        .iter()
        .map(|&point| {
            let relative = point - outlet;
            (relative.dot(along), relative.dot(transverse))
        })
        .collect();
    let polyline_arclength_km: f64 = vertices
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum();
    let total_backtracking_km: f64 = coordinates
        .windows(2)
        .map(|pair| (pair[1].0 - pair[0].0).max(0.0))
        .sum();
    let (terminal_along_track_km, terminal_cross_track_km) = coordinates[coordinates.len() - 1];
    CrossingMetrics {
        maximum_absolute_cross_track_km: coordinates
            .iter()
            .map(|(_, n)| n.abs())
            .fold(0.0, f64::max),
        polyline_arclength_km,
        relative_arclength_error: (polyline_arclength_km - R1_HEAD_ALONG_TRACK_KM).abs()
            / R1_HEAD_ALONG_TRACK_KM,
        total_backtracking_km,
        endpoint_error_km: vertices[vertices.len() - 1].distance(outlet),
        terminal_along_track_km,
        terminal_cross_track_km,
    }
}

fn observe_pair(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    context: &CheckedSegmentContext,
    reconstruction: &ReconstructionAudit,
) -> Result<PairedCrossingObservation, R1CaseError> {
    let x0 = trace_crossing(
        cap,
        case,
        context,
        reconstruction,
        CrossingArm::X0Analytic,
        R1_CAP_PORTAL_ID,
    )?;
    let x1 = reconstruction
        .passes_gate()
        .then(|| {
            trace_crossing(
                cap,
                case,
                context,
                reconstruction,
                CrossingArm::X1CentroidReconstruction,
                R1_CAP_PORTAL_ID,
            )
        })
        .transpose()?;
    let face_sequences_equal = x1.as_ref().map(|x1| {
        x0.crossings
            .iter()
            .map(|crossing| crossing.face)
            .eq(x1.crossings.iter().map(|crossing| crossing.face))
    });
    let endpoint_difference_km = x1.as_ref().map(|x1| {
        x0.vertices_km[x0.vertices_km.len() - 1].distance(x1.vertices_km[x1.vertices_km.len() - 1])
    });
    let per_crossing_maximum_difference_km = match (&x1, face_sequences_equal) {
        (Some(x1), Some(true)) => Some(
            x0.crossings
                .iter()
                .zip(&x1.crossings)
                .map(|(x0, x1)| x0.point_km.distance(x1.point_km))
                .fold(0.0, f64::max),
        ),
        _ => None,
    };
    let metric_differences = x1.as_ref().and_then(|x1| {
        let (Some(x0), Some(x1)) = (&x0.metrics, &x1.metrics) else {
            return None;
        };
        Some(CrossingMetricDifferences {
            maximum_absolute_cross_track_km: x1.maximum_absolute_cross_track_km
                - x0.maximum_absolute_cross_track_km,
            polyline_arclength_km: x1.polyline_arclength_km - x0.polyline_arclength_km,
            relative_arclength_error: x1.relative_arclength_error - x0.relative_arclength_error,
            total_backtracking_km: x1.total_backtracking_km - x0.total_backtracking_km,
            endpoint_error_km: x1.endpoint_error_km - x0.endpoint_error_km,
            terminal_along_track_km: x1.terminal_along_track_km - x0.terminal_along_track_km,
            terminal_cross_track_km: x1.terminal_cross_track_km - x0.terminal_cross_track_km,
        })
    });
    Ok(PairedCrossingObservation {
        x0,
        x1,
        face_sequences_equal,
        endpoint_difference_km,
        per_crossing_maximum_difference_km,
        metric_differences,
    })
}

fn validate_affine_case(
    cap: &VoronoiCapFixture,
    case: &R1RegisteredCase,
    context: &CheckedSegmentContext,
) -> Result<(), R1CaseError> {
    if case.config.surface != R1SurfaceKind::Affine
        || case.config.spacing_km != cap.config.spacing_km
        || context.spacing_km != cap.config.spacing_km
        || context.cells.len() != cap.mesh.cell_count()
        || case.elevation_km.len() != cap.mesh.cell_count()
        || case.heads.len() != 1
    {
        return Err(R1CaseError(
            "affine crossing inputs are incompatible or out of scope".into(),
        ));
    }
    Ok(())
}

fn case_frame(config: R1CaseConfig) -> (DVec2, DVec2, DVec2) {
    let along = DVec2::new(config.theta_rad.sin(), config.theta_rad.cos());
    let transverse = DVec2::new(-config.theta_rad.cos(), config.theta_rad.sin());
    (
        DVec2::new(config.outlet_offset_km, -0.5 * R1_CAP_HEIGHT_KM),
        along,
        transverse,
    )
}

fn registered_case(
    cap: &VoronoiCapFixture,
    theta_rad: f64,
    outlet_offset_km: f64,
) -> R1RegisteredCase {
    build_r1_registered_case(
        cap,
        R1CaseConfig {
            spacing_km: cap.config.spacing_km,
            theta_rad,
            outlet_offset_km,
            surface: R1SurfaceKind::Affine,
        },
    )
    .unwrap()
}

#[test]
fn checked_context_reconstruction_and_trace_are_deterministic_and_immutable() {
    let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
    let case = registered_case(&cap, 0.31, 0.7);
    let cap_before = cap.clone();
    let case_before = case.clone();
    let first_context = CheckedSegmentContext::new(&cap).unwrap();
    let second_context = CheckedSegmentContext::new(&cap).unwrap();
    assert_eq!(first_context, second_context);
    assert_eq!(
        first_context.edge_segment.len(),
        cap.mesh.edge_neighbor.len()
    );
    assert_eq!(
        first_context.boundary_segment.len(),
        cap.mesh.boundary_faces.len()
    );

    let first_reconstruction = reconstruct_centroid_gradients(&cap, &case, &first_context).unwrap();
    let second_reconstruction =
        reconstruct_centroid_gradients(&cap, &case, &first_context).unwrap();
    assert_eq!(first_reconstruction, second_reconstruction);
    let first_scores =
        audit_internal_score_equivalence(&cap, &case, &first_reconstruction).unwrap();
    let second_scores =
        audit_internal_score_equivalence(&cap, &case, &first_reconstruction).unwrap();
    assert_eq!(first_scores, second_scores);

    let first = observe_pair(&cap, &case, &first_context, &first_reconstruction).unwrap();
    let second = observe_pair(&cap, &case, &first_context, &first_reconstruction).unwrap();
    assert_eq!(first, second);
    for trace in std::iter::once(&first.x0).chain(first.x1.iter()) {
        for crossing in &trace.crossings {
            assert!(crossing.ray_parameter_km > RELATIVE_TOLERANCE * cap.config.spacing_km);
            assert!(crossing.segment_parameter >= -RELATIVE_TOLERANCE);
            assert!(crossing.segment_parameter <= 1.0 + RELATIVE_TOLERANCE);
            assert!(crossing.segment_residual_km <= RELATIVE_TOLERANCE * cap.config.spacing_km);
            assert!(crossing
                .reciprocal_segment_residual_km
                .is_none_or(|residual| residual <= RELATIVE_TOLERANCE * cap.config.spacing_km));
        }
    }
    assert_eq!(cap, cap_before);
    assert_eq!(case, case_before);
    assert_eq!(first_context, second_context);
}

#[test]
fn synthetic_metrics_use_complete_crossing_polyline() {
    let config = R1CaseConfig {
        spacing_km: 8.0,
        theta_rad: 0.0,
        outlet_offset_km: 0.0,
        surface: R1SurfaceKind::Affine,
    };
    let vertices = [
        DVec2::new(0.0, 64.0),
        DVec2::new(3.0, 20.0),
        DVec2::new(0.0, 24.0),
        DVec2::new(0.0, -112.0),
    ];
    let metrics = crossing_metrics(&vertices, config);
    assert_eq!(metrics.maximum_absolute_cross_track_km, 3.0);
    assert_eq!(metrics.total_backtracking_km, 4.0);
    assert_eq!(metrics.endpoint_error_km, 0.0);
    assert_eq!(metrics.terminal_along_track_km, 0.0);
    assert_eq!(metrics.terminal_cross_track_km, 0.0);
    let expected_length = 44.0_f64.hypot(3.0) + 5.0 + 136.0;
    assert_eq!(metrics.polyline_arclength_km, expected_length);
    assert_eq!(
        metrics.relative_arclength_error,
        (expected_length - 176.0).abs() / 176.0
    );
}

fn metric_range(values: impl Iterator<Item = f64>) -> Option<(f64, f64, f64)> {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut count = 0usize;
    for value in values {
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        count += 1;
    }
    (count > 0).then_some((minimum, maximum, maximum - minimum))
}

fn report_spacing_summary(spacing_km: f64, observations: &[PairedCrossingObservation]) {
    for arm in [
        CrossingArm::X0Analytic,
        CrossingArm::X1CentroidReconstruction,
    ] {
        let traces: Vec<_> = observations
            .iter()
            .filter_map(|observation| match arm {
                CrossingArm::X0Analytic => Some(&observation.x0),
                CrossingArm::X1CentroidReconstruction => observation.x1.as_ref(),
            })
            .collect();
        let successful: Vec<_> = traces
            .iter()
            .filter_map(|trace| trace.metrics.as_ref())
            .collect();
        let reconstruction_not_judged_count = match arm {
            CrossingArm::X0Analytic => 0,
            CrossingArm::X1CentroidReconstruction => observations
                .iter()
                .filter(|observation| observation.x1.is_none())
                .count(),
        };
        let traversal_failure_count = traces
            .iter()
            .filter(|trace| trace.metrics.is_none())
            .count();
        println!(
            "spacing_summary={spacing_km} arm={arm:?} success_count={} reconstruction_not_judged_count={reconstruction_not_judged_count} traversal_failure_count={traversal_failure_count} cross={:?} length={:?} relative_length={:?} backtracking={:?} endpoint={:?} terminal_s={:?} terminal_n={:?}",
            successful.len(),
            metric_range(successful.iter().map(|metrics| metrics.maximum_absolute_cross_track_km)),
            metric_range(successful.iter().map(|metrics| metrics.polyline_arclength_km)),
            metric_range(successful.iter().map(|metrics| metrics.relative_arclength_error)),
            metric_range(successful.iter().map(|metrics| metrics.total_backtracking_km)),
            metric_range(successful.iter().map(|metrics| metrics.endpoint_error_km)),
            metric_range(successful.iter().map(|metrics| metrics.terminal_along_track_km)),
            metric_range(successful.iter().map(|metrics| metrics.terminal_cross_track_km)),
        );
    }
}

fn passes_frozen_trace_gate(trace: &CrossingTrace, spacing_km: f64) -> bool {
    let Some(metrics) = trace.metrics.as_ref() else {
        return false;
    };
    let tau_x = RELATIVE_TOLERANCE * spacing_km;
    let backtracking_tolerance = match trace.arm {
        CrossingArm::X0Analytic => tau_x,
        CrossingArm::X1CentroidReconstruction => (trace.crossings.len() + 1) as f64 * tau_x,
    };
    trace.termination == CrossingTermination::ReachedPortal
        && metrics.maximum_absolute_cross_track_km <= tau_x
        && metrics.total_backtracking_km <= backtracking_tolerance
        && metrics.relative_arclength_error < 0.05
}

#[test]
#[ignore = "full registered affine X0/X1 crossing matrix is an audit test"]
fn registered_affine_crossing_reproduces_frozen_incomplete_matrix() {
    let mut case_count = 0usize;
    let mut x0_pass_count = 0usize;
    let mut reconstruction_pass_count = 0usize;
    let mut x1_pass_count = 0usize;
    let mut score_pass_count = 0usize;
    let mut equal_face_sequence_count = 0usize;
    let mut reconstruction_failures = Vec::new();
    for spacing_km in [8.0, 4.0, 2.0] {
        let cap = build_r1_voronoi_cap(VoronoiCapConfig::r1(spacing_km)).unwrap();
        let cap_before = cap.clone();
        let context = CheckedSegmentContext::new(&cap).unwrap();
        assert_eq!(context, CheckedSegmentContext::new(&cap).unwrap());
        let mut observations = Vec::new();
        for theta_rad in [0.0, 0.31] {
            for outlet_offset_km in [0.0, 0.7] {
                let case = registered_case(&cap, theta_rad, outlet_offset_km);
                let case_before = case.clone();
                assert_eq!(case, registered_case(&cap, theta_rad, outlet_offset_km));
                let reconstruction = reconstruct_centroid_gradients(&cap, &case, &context).unwrap();
                assert_eq!(
                    reconstruction,
                    reconstruct_centroid_gradients(&cap, &case, &context).unwrap()
                );
                let scores =
                    audit_internal_score_equivalence(&cap, &case, &reconstruction).unwrap();
                assert_eq!(
                    scores,
                    audit_internal_score_equivalence(&cap, &case, &reconstruction).unwrap()
                );
                let observation = observe_pair(&cap, &case, &context, &reconstruction).unwrap();
                assert_eq!(
                    observation,
                    observe_pair(&cap, &case, &context, &reconstruction).unwrap()
                );
                println!(
                    "spacing={spacing_km} theta={theta_rad} delta={outlet_offset_km} reconstruction_rel={:.6e} reconstruction_sine={:.6e} reconstruction_gate={} eligibility_conflicts={} winner_conflicts={} score_gate={} min_gen_margin={:?} min_x_margin={:?} min_abs_sgen={:.6e} max_score_rel={:.6e} x0_term={:?} x0_cells={} x0_faces={} x0_metrics={:?} x0_segment_residual={:.6e} x0_vertex_clearance={:?} x0_exit_gap={:?} x0_visited={:?} x0_crossings={:?} x1_term={:?} x1_cells={:?} x1_faces={:?} x1_metrics={:?} x1_segment_residual={:?} x1_vertex_clearance={:?} x1_exit_gap={:?} x1_visited_gradient_rel={:?} x1_visited_gradient_sine={:?} x1_visited={:?} x1_crossings={:?} faces_equal={:?} endpoint_delta={:?} crossing_delta={:?} metric_differences={:?}",
                    reconstruction.maximum_relative_vector_error,
                    reconstruction.maximum_sine_angle_error,
                    reconstruction.passes_gate(),
                    scores.eligibility_conflicts,
                    scores.winner_conflicts,
                    scores.passes_gate(),
                    scores.minimum_generator_normalized_margin,
                    scores.minimum_reconstructed_normalized_margin,
                    scores.minimum_absolute_generator_score,
                    scores.maximum_symmetric_normalized_score_error,
                    observation.x0.termination,
                    observation.x0.visited_cells.len(),
                    observation.x0.crossings.len(),
                    observation.x0.metrics,
                    observation.x0.maximum_segment_residual_km,
                    observation.x0.minimum_vertex_clearance_km,
                    observation.x0.minimum_exit_parameter_gap_km,
                    observation.x0.visited_cells,
                    observation.x0.crossings,
                    observation.x1.as_ref().map(|trace| trace.termination),
                    observation.x1.as_ref().map(|trace| trace.visited_cells.len()),
                    observation.x1.as_ref().map(|trace| trace.crossings.len()),
                    observation.x1.as_ref().and_then(|trace| trace.metrics.as_ref()),
                    observation
                        .x1
                        .as_ref()
                        .map(|trace| trace.maximum_segment_residual_km),
                    observation
                        .x1
                        .as_ref()
                        .and_then(|trace| trace.minimum_vertex_clearance_km),
                    observation
                        .x1
                        .as_ref()
                        .and_then(|trace| trace.minimum_exit_parameter_gap_km),
                    observation
                        .x1
                        .as_ref()
                        .and_then(|trace| trace.visited_maximum_relative_gradient_error),
                    observation
                        .x1
                        .as_ref()
                        .and_then(|trace| trace.visited_maximum_sine_angle_error),
                    observation.x1.as_ref().map(|trace| &trace.visited_cells),
                    observation.x1.as_ref().map(|trace| &trace.crossings),
                    observation.face_sequences_equal,
                    observation.endpoint_difference_km,
                    observation.per_crossing_maximum_difference_km,
                    observation.metric_differences,
                );
                case_count += 1;
                x0_pass_count += usize::from(passes_frozen_trace_gate(&observation.x0, spacing_km));
                reconstruction_pass_count += usize::from(reconstruction.passes_gate());
                score_pass_count += usize::from(scores.passes_gate());
                if !reconstruction.passes_gate() {
                    reconstruction_failures.push((spacing_km, theta_rad, outlet_offset_km));
                }
                if let Some(x1) = observation.x1.as_ref() {
                    x1_pass_count += usize::from(passes_frozen_trace_gate(x1, spacing_km));
                }
                equal_face_sequence_count +=
                    usize::from(observation.face_sequences_equal == Some(true));
                assert_eq!(case, case_before);
                observations.push(observation);
            }
        }
        report_spacing_summary(spacing_km, &observations);
        assert_eq!(cap, cap_before);
        assert_eq!(context, CheckedSegmentContext::new(&cap).unwrap());
    }
    assert_eq!(case_count, 12);
    assert_eq!(x0_pass_count, 12);
    assert_eq!(score_pass_count, 12);
    assert_eq!(reconstruction_pass_count, 11);
    assert_eq!(x1_pass_count, 11);
    assert_eq!(equal_face_sequence_count, 11);
    assert_eq!(reconstruction_failures, [(2.0, 0.31, 0.0)]);
}
