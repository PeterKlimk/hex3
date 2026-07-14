//! Global planar-subdivision validation for O0a relationship evidence.
//!
//! G0 validates each control volume and its face backing locally.  O0a also
//! performs point location and ray traversal, so it needs the stronger global
//! fact that those control volumes form a non-overlapping planar subdivision.

use bincode::Options;
use glam::{DVec2, DVec3};

use super::super::landscape::{build_r1_voronoi_cap, VoronoiCapConfig};
use super::{
    adapt_projected_voronoi_cap_graph_v0, EvaluationDomainV0, EvaluationSurfaceGraphV0,
    RelationshipErrorV0, SurfaceHierarchyConfigV0,
};

#[derive(Clone, Copy)]
struct Bounds {
    min: DVec2,
    max: DVec2,
}

impl Bounds {
    fn of(polygon: &[DVec2]) -> Self {
        let mut min = polygon[0];
        let mut max = polygon[0];
        for &point in &polygon[1..] {
            min = min.min(point);
            max = max.max(point);
        }
        Self { min, max }
    }

    fn overlaps(self, other: Self, tolerance: f64) -> bool {
        self.max.x + tolerance >= other.min.x
            && other.max.x + tolerance >= self.min.x
            && self.max.y + tolerance >= other.min.y
            && other.max.y + tolerance >= self.min.y
    }
}

/// Certify the sole registered projected O0a geometry identity.
///
/// A caller-provided hash can establish that a label describes its input, but
/// cannot establish that the input is the registered R1 control.  Rebuilding
/// that manufactured control and requiring full graph equality prevents an
/// arbitrary planar graph from self-labeling as projected R1 evidence.
pub(super) fn validate_projected_r1_voronoi_cap_identity_v0(
    graph: &EvaluationSurfaceGraphV0,
    expected_graph_hash: u64,
) -> Result<(), RelationshipErrorV0> {
    let config = SurfaceHierarchyConfigV0::default();
    let fixture = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0))
        .map_err(|_| RelationshipErrorV0::InvalidGeometryIdentity)?;
    let registered = adapt_projected_voronoi_cap_graph_v0(&fixture, &config)
        .map_err(|_| RelationshipErrorV0::InvalidGeometryIdentity)?;
    if graph != &registered || canonical_graph_hash(&registered)? != expected_graph_hash {
        return Err(RelationshipErrorV0::InvalidGeometryIdentity);
    }
    Ok(())
}

/// Validate the global planar-subdivision properties required by O0a.
///
/// The registered planar control volumes are convex.  Convexity is checked
/// explicitly before the convex-clipping overlap test is used.  Boundary
/// segments retain their owning polygons' direction, so outer loops contribute
/// positive signed area and hole loops contribute negative signed area.
pub(super) fn validate_planar_subdivision_v0(
    graph: &EvaluationSurfaceGraphV0,
    endpoint_tol: f64,
    area_rel_tol: f64,
) -> Result<(), RelationshipErrorV0> {
    if graph.domain != EvaluationDomainV0::Planar {
        return Err(invalid(
            "planar subdivision validation requires a planar graph",
        ));
    }
    if !endpoint_tol.is_finite()
        || endpoint_tol < 0.0
        || !area_rel_tol.is_finite()
        || area_rel_tol < 0.0
    {
        return Err(invalid("invalid planar subdivision tolerance"));
    }

    let mut polygons = Vec::with_capacity(graph.cell_count());
    let mut perimeters = Vec::with_capacity(graph.cell_count());
    let mut bounds = Vec::with_capacity(graph.cell_count());
    for cell in 0..graph.cell_count() {
        let polygon: Vec<DVec2> = graph
            .polygon(cell)
            .iter()
            .map(|point| DVec2::new(point.x, point.y))
            .collect();
        validate_cell_polygon(cell, &polygon, graph.cell_center_km[cell], endpoint_tol)?;
        perimeters.push(polygon_perimeter(&polygon));
        bounds.push(Bounds::of(&polygon));
        polygons.push(polygon);
    }

    // Convex clipping is deterministic and adequate only because convexity was
    // established above.  A deterministic x sweep limits pair tests to cells
    // whose bounding boxes can overlap.
    let mut sweep_order: Vec<usize> = (0..polygons.len()).collect();
    sweep_order.sort_by(|&first, &second| {
        bounds[first]
            .min
            .x
            .total_cmp(&bounds[second].min.x)
            .then_with(|| first.cmp(&second))
    });
    let mut active = Vec::<usize>::new();
    for second in sweep_order {
        active.retain(|&first| bounds[first].max.x + endpoint_tol >= bounds[second].min.x);
        for &first in &active {
            if !bounds[first].overlaps(bounds[second], endpoint_tol) {
                continue;
            }
            let overlap = convex_intersection_area(&polygons[first], &polygons[second]);
            let allowed = endpoint_tol * (perimeters[first] + perimeters[second]);
            if !overlap.is_finite() || overlap > allowed {
                return Err(invalid(format!(
                    "cell polygons {first} and {second} overlap by {overlap} km^2"
                )));
            }
        }
        active.push(second);
    }

    let signed_boundary_area = validate_boundary_loops(graph, endpoint_tol)?;
    let cell_area = compensated_sum(graph.cell_area_km2.iter().copied());
    if !signed_boundary_area.is_finite() || signed_boundary_area <= 0.0 {
        return Err(invalid(
            "planar boundary loops have non-positive signed area",
        ));
    }
    let relative =
        (cell_area - signed_boundary_area).abs() / cell_area.abs().max(f64::MIN_POSITIVE);
    if !relative.is_finite() || relative > area_rel_tol {
        return Err(invalid(format!(
            "cell area and signed boundary-loop area differ by relative {relative}"
        )));
    }
    Ok(())
}

fn validate_cell_polygon(
    cell: usize,
    polygon: &[DVec2],
    center: DVec3,
    tolerance: f64,
) -> Result<(), RelationshipErrorV0> {
    if polygon.len() < 3 || polygon.iter().any(|point| !point.is_finite()) {
        return Err(invalid(format!("cell {cell} has an invalid polygon")));
    }
    let center = DVec2::new(center.x, center.y);
    if !center.is_finite() {
        return Err(invalid(format!("cell {cell} has a non-finite center")));
    }

    let edges: Vec<(DVec2, DVec2)> = polygon
        .iter()
        .copied()
        .zip(polygon.iter().copied().cycle().skip(1))
        .take(polygon.len())
        .collect();
    for (edge, &(start, end)) in edges.iter().enumerate() {
        let vector = end - start;
        let length = vector.length();
        if !length.is_finite() || length <= tolerance {
            return Err(invalid(format!("cell {cell} has degenerate edge {edge}")));
        }
        // The source polygons are CCW.  A tolerance-wide strip along an edge is
        // considered boundary, not strict interior.
        if cross(vector, center - start) <= tolerance * length {
            return Err(invalid(format!(
                "cell {cell} center is not strictly inside its polygon"
            )));
        }
    }

    for edge in 0..edges.len() {
        let next = (edge + 1) % edges.len();
        let incoming = edges[edge].1 - edges[edge].0;
        let outgoing = edges[next].1 - edges[next].0;
        let turn_tolerance = tolerance * (incoming.length() + outgoing.length());
        if cross(incoming, outgoing) < -turn_tolerance {
            return Err(invalid(format!("cell {cell} polygon is not convex")));
        }
    }

    for first in 0..edges.len() {
        for second in first + 1..edges.len() {
            let adjacent = second == first + 1 || (first == 0 && second + 1 == edges.len());
            if segments_intersect_disallowed(edges[first], edges[second], tolerance, adjacent) {
                return Err(invalid(format!("cell {cell} polygon is not simple")));
            }
        }
    }
    Ok(())
}

fn validate_boundary_loops(
    graph: &EvaluationSurfaceGraphV0,
    tolerance: f64,
) -> Result<f64, RelationshipErrorV0> {
    let segments: Vec<(DVec2, DVec2)> = graph
        .boundary_segments
        .iter()
        .map(|segment| {
            (
                DVec2::new(segment.endpoints_km[0].x, segment.endpoints_km[0].y),
                DVec2::new(segment.endpoints_km[1].x, segment.endpoints_km[1].y),
            )
        })
        .collect();
    if segments.is_empty() {
        return Err(invalid("planar subdivision has no boundary segments"));
    }
    if segments.iter().any(|&(start, end)| {
        !start.is_finite() || !end.is_finite() || start.distance(end) <= tolerance
    }) {
        return Err(invalid("planar boundary contains a degenerate segment"));
    }

    let mut next = vec![usize::MAX; segments.len()];
    let mut predecessor_count = vec![0usize; segments.len()];
    for (index, &(_, end)) in segments.iter().enumerate() {
        let matches: Vec<usize> = segments
            .iter()
            .enumerate()
            .filter_map(|(candidate, &(start, _))| {
                (end.distance(start) <= tolerance).then_some(candidate)
            })
            .collect();
        if matches.len() != 1 {
            return Err(invalid(format!(
                "boundary segment {index} has {} directed successors",
                matches.len()
            )));
        }
        next[index] = matches[0];
        predecessor_count[matches[0]] += 1;
    }
    if predecessor_count.iter().any(|&count| count != 1) {
        return Err(invalid("boundary segments do not have unique predecessors"));
    }

    let mut visited = vec![false; segments.len()];
    for start in 0..segments.len() {
        if visited[start] {
            continue;
        }
        let mut current = start;
        loop {
            if visited[current] {
                if current != start {
                    return Err(invalid("boundary chain joins a different loop"));
                }
                break;
            }
            visited[current] = true;
            current = next[current];
        }
    }

    for first in 0..segments.len() {
        for second in first + 1..segments.len() {
            let adjacent = next[first] == second || next[second] == first;
            if segments_intersect_disallowed(segments[first], segments[second], tolerance, adjacent)
            {
                return Err(invalid(format!(
                    "boundary segments {first} and {second} intersect"
                )));
            }
        }
    }

    Ok(0.5 * compensated_sum(segments.iter().map(|&(start, end)| cross(start, end))))
}

fn convex_intersection_area(subject: &[DVec2], clip: &[DVec2]) -> f64 {
    let mut output = subject.to_vec();
    for (&clip_start, &clip_end) in clip
        .iter()
        .zip(clip.iter().cycle().skip(1))
        .take(clip.len())
    {
        if output.is_empty() {
            return 0.0;
        }
        let input = std::mem::take(&mut output);
        let mut previous = *input.last().expect("non-empty clipping input");
        let mut previous_side = cross(clip_end - clip_start, previous - clip_start);
        for current in input {
            let current_side = cross(clip_end - clip_start, current - clip_start);
            let previous_inside = previous_side >= 0.0;
            let current_inside = current_side >= 0.0;
            if previous_inside != current_inside {
                let denominator = previous_side - current_side;
                if denominator != 0.0 {
                    output.push(previous + (current - previous) * (previous_side / denominator));
                }
            }
            if current_inside {
                output.push(current);
            }
            previous = current;
            previous_side = current_side;
        }
    }
    polygon_signed_area(&output).abs()
}

fn segments_intersect_disallowed(
    first: (DVec2, DVec2),
    second: (DVec2, DVec2),
    tolerance: f64,
    adjacent: bool,
) -> bool {
    let (a, b) = first;
    let (c, d) = second;
    let ab = b - a;
    let cd = d - c;
    let scale = ab.length() + cd.length();
    let cross_tolerance = tolerance * scale;
    let o1 = cross(ab, c - a);
    let o2 = cross(ab, d - a);
    let o3 = cross(cd, a - c);
    let o4 = cross(cd, b - c);

    let proper = ((o1 > cross_tolerance && o2 < -cross_tolerance)
        || (o1 < -cross_tolerance && o2 > cross_tolerance))
        && ((o3 > cross_tolerance && o4 < -cross_tolerance)
            || (o3 < -cross_tolerance && o4 > cross_tolerance));
    if proper {
        return true;
    }

    let contacts = point_on_segment(c, a, b, tolerance)
        || point_on_segment(d, a, b, tolerance)
        || point_on_segment(a, c, d, tolerance)
        || point_on_segment(b, c, d, tolerance);
    if !contacts {
        return false;
    }
    if !adjacent {
        return true;
    }

    // Consecutive segments may meet at their shared endpoint.  Collinear
    // continuation is also valid, but any positive-length backtracking overlap
    // is a self-intersection.
    if cross(ab, cd).abs() > cross_tolerance {
        return false;
    }
    collinear_overlap_length(a, b, c, d) > tolerance
}

fn point_on_segment(point: DVec2, start: DVec2, end: DVec2, tolerance: f64) -> bool {
    let edge = end - start;
    let length = edge.length();
    if cross(edge, point - start).abs() > tolerance * length {
        return false;
    }
    let projection = (point - start).dot(edge) / length;
    projection >= -tolerance && projection <= length + tolerance
}

fn collinear_overlap_length(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> f64 {
    let direction = (b - a).normalize();
    let first = [0.0, (b - a).dot(direction)];
    let second = [(c - a).dot(direction), (d - a).dot(direction)];
    let first_min = first[0].min(first[1]);
    let first_max = first[0].max(first[1]);
    let second_min = second[0].min(second[1]);
    let second_max = second[0].max(second[1]);
    (first_max.min(second_max) - first_min.max(second_min)).max(0.0)
}

fn polygon_perimeter(polygon: &[DVec2]) -> f64 {
    compensated_sum(
        polygon
            .iter()
            .copied()
            .zip(polygon.iter().copied().cycle().skip(1))
            .take(polygon.len())
            .map(|(start, end)| start.distance(end)),
    )
}

fn polygon_signed_area(polygon: &[DVec2]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    0.5 * compensated_sum(
        polygon
            .iter()
            .copied()
            .zip(polygon.iter().copied().cycle().skip(1))
            .take(polygon.len())
            .map(|(start, end)| cross(start, end)),
    )
}

fn cross(a: DVec2, b: DVec2) -> f64 {
    a.x * b.y - a.y * b.x
}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn invalid(message: impl Into<String>) -> RelationshipErrorV0 {
    RelationshipErrorV0::InvalidGraph(message.into())
}

fn canonical_graph_hash(graph: &EvaluationSurfaceGraphV0) -> Result<u64, RelationshipErrorV0> {
    let bytes = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_little_endian()
        .serialize(graph)
        .map_err(|error| RelationshipErrorV0::Serialization(error.to_string()))?;
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::landforms::{
        adapt_landscape_graph_v0, build_regular_hex_control_volumes_v0,
        EvaluationBoundaryConditionV0, EvaluationBoundarySegmentV0,
    };
    use crate::world::landscape::LandscapeMesh;

    fn point(x: f64, y: f64) -> DVec3 {
        DVec3::new(x, y, 0.0)
    }

    fn graph(
        polygons: &[Vec<DVec3>],
        centers: &[DVec3],
        boundary: &[[DVec3; 2]],
    ) -> EvaluationSurfaceGraphV0 {
        let mut offsets = Vec::with_capacity(polygons.len() + 1);
        let mut vertices = Vec::new();
        let mut areas = Vec::new();
        for polygon in polygons {
            offsets.push(vertices.len() as u32);
            vertices.extend(polygon);
            let projected: Vec<DVec2> = polygon.iter().map(|p| DVec2::new(p.x, p.y)).collect();
            areas.push(polygon_signed_area(&projected));
        }
        offsets.push(vertices.len() as u32);
        let boundary_segments = boundary
            .iter()
            .enumerate()
            .map(|(id, &endpoints_km)| EvaluationBoundarySegmentV0 {
                id: id as u32,
                owner_cell: 0,
                endpoints_km,
                physical_length_km: endpoints_km[0].distance(endpoints_km[1]),
                projected_span_km: None,
                condition: EvaluationBoundaryConditionV0::Closed,
            })
            .collect();
        EvaluationSurfaceGraphV0 {
            domain: EvaluationDomainV0::Planar,
            cell_center_km: centers.to_vec(),
            cell_area_km2: areas,
            cell_polygon_offsets: offsets,
            cell_polygon_vertices_km: vertices,
            edge_offsets: vec![0; polygons.len() + 1],
            edge_neighbor: Vec::new(),
            edge_reciprocal: Vec::new(),
            edge_distance_km: Vec::new(),
            edge_shared_width_km: Vec::new(),
            edge_face_endpoints_km: Vec::new(),
            boundary_segments,
        }
    }

    #[test]
    fn accepts_two_cell_rectangular_subdivision() {
        let polygons = vec![
            vec![
                point(0.0, 0.0),
                point(1.0, 0.0),
                point(1.0, 1.0),
                point(0.0, 1.0),
            ],
            vec![
                point(1.0, 0.0),
                point(2.0, 0.0),
                point(2.0, 1.0),
                point(1.0, 1.0),
            ],
        ];
        let boundary = [
            [point(0.0, 0.0), point(1.0, 0.0)],
            [point(1.0, 0.0), point(2.0, 0.0)],
            [point(2.0, 0.0), point(2.0, 1.0)],
            [point(2.0, 1.0), point(1.0, 1.0)],
            [point(1.0, 1.0), point(0.0, 1.0)],
            [point(0.0, 1.0), point(0.0, 0.0)],
        ];
        let graph = graph(&polygons, &[point(0.5, 0.5), point(1.5, 0.5)], &boundary);
        assert!(validate_planar_subdivision_v0(&graph, 1.0e-9, 1.0e-9).is_ok());
    }

    #[test]
    fn accepts_registered_irregular_r1_voronoi_cap() {
        let config = SurfaceHierarchyConfigV0::default();
        let fixture = build_r1_voronoi_cap(VoronoiCapConfig::r1(8.0)).unwrap();
        let graph = adapt_projected_voronoi_cap_graph_v0(&fixture, &config).unwrap();
        validate_planar_subdivision_v0(
            &graph,
            config.endpoint_match_abs_km,
            config.planar_area_match_relative,
        )
        .unwrap();
        let graph_hash = canonical_graph_hash(&graph).unwrap();
        validate_projected_r1_voronoi_cap_identity_v0(&graph, graph_hash).unwrap();
    }

    #[test]
    fn regular_graph_cannot_self_label_as_projected_r1() {
        let config = SurfaceHierarchyConfigV0::default();
        let mesh = LandscapeMesh::uniform_planar_hex_with_portals(48.0, 40.0, 4.0, &[]).unwrap();
        let controls = build_regular_hex_control_volumes_v0(&mesh, &config).unwrap();
        let graph = adapt_landscape_graph_v0(&mesh, &controls, &config).unwrap();
        let self_hash = canonical_graph_hash(&graph).unwrap();
        assert_eq!(
            validate_projected_r1_voronoi_cap_identity_v0(&graph, self_hash),
            Err(RelationshipErrorV0::InvalidGeometryIdentity)
        );
    }

    #[test]
    fn rejects_positive_area_cell_overlap() {
        let polygons = vec![
            vec![
                point(0.0, 0.0),
                point(1.1, 0.0),
                point(1.1, 1.0),
                point(0.0, 1.0),
            ],
            vec![
                point(1.0, 0.0),
                point(2.0, 0.0),
                point(2.0, 1.0),
                point(1.0, 1.0),
            ],
        ];
        let boundary = [
            [point(0.0, 0.0), point(2.0, 0.0)],
            [point(2.0, 0.0), point(2.0, 1.0)],
            [point(2.0, 1.0), point(0.0, 1.0)],
            [point(0.0, 1.0), point(0.0, 0.0)],
        ];
        let graph = graph(&polygons, &[point(0.5, 0.5), point(1.5, 0.5)], &boundary);
        assert!(validate_planar_subdivision_v0(&graph, 1.0e-9, 1.0e-9).is_err());
    }

    #[test]
    fn rejects_undeclared_internal_gap_by_area_closure() {
        let polygons = vec![
            vec![
                point(0.0, 0.0),
                point(0.9, 0.0),
                point(0.9, 1.0),
                point(0.0, 1.0),
            ],
            vec![
                point(1.1, 0.0),
                point(2.0, 0.0),
                point(2.0, 1.0),
                point(1.1, 1.0),
            ],
        ];
        let boundary = [
            [point(0.0, 0.0), point(2.0, 0.0)],
            [point(2.0, 0.0), point(2.0, 1.0)],
            [point(2.0, 1.0), point(0.0, 1.0)],
            [point(0.0, 1.0), point(0.0, 0.0)],
        ];
        let graph = graph(&polygons, &[point(0.45, 0.5), point(1.55, 0.5)], &boundary);
        assert!(validate_planar_subdivision_v0(&graph, 1.0e-9, 1.0e-9).is_err());
    }

    #[test]
    fn rejects_center_on_polygon_boundary() {
        let polygon = vec![
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(1.0, 1.0),
            point(0.0, 1.0),
        ];
        let boundary = [
            [point(0.0, 0.0), point(1.0, 0.0)],
            [point(1.0, 0.0), point(1.0, 1.0)],
            [point(1.0, 1.0), point(0.0, 1.0)],
            [point(0.0, 1.0), point(0.0, 0.0)],
        ];
        let graph = graph(&[polygon], &[point(0.0, 0.5)], &boundary);
        assert!(validate_planar_subdivision_v0(&graph, 1.0e-9, 1.0e-9).is_err());
    }

    #[test]
    fn rejects_open_boundary_chain() {
        let polygon = vec![
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(1.0, 1.0),
            point(0.0, 1.0),
        ];
        let boundary = [
            [point(0.0, 0.0), point(1.0, 0.0)],
            [point(1.0, 0.0), point(1.0, 1.0)],
            [point(1.0, 1.0), point(0.0, 1.0)],
        ];
        let graph = graph(&[polygon], &[point(0.5, 0.5)], &boundary);
        assert!(validate_planar_subdivision_v0(&graph, 1.0e-9, 1.0e-9).is_err());
    }

    #[test]
    fn rejects_closed_self_intersecting_boundary_loop() {
        let polygon = vec![
            point(0.0, 0.0),
            point(1.0, 0.0),
            point(1.0, 1.0),
            point(0.0, 1.0),
        ];
        let boundary = [
            [point(0.0, 0.0), point(1.0, 1.0)],
            [point(1.0, 1.0), point(0.0, 1.0)],
            [point(0.0, 1.0), point(1.0, 0.0)],
            [point(1.0, 0.0), point(0.0, 0.0)],
        ];
        let graph = graph(&[polygon], &[point(0.5, 0.5)], &boundary);
        assert!(validate_planar_subdivision_v0(&graph, 1.0e-9, 1.0e-9).is_err());
    }

    #[test]
    fn rejects_non_convex_cell_before_clipping() {
        let polygon = vec![
            point(0.0, 0.0),
            point(2.0, 0.0),
            point(2.0, 1.0),
            point(1.0, 1.0),
            point(1.0, 2.0),
            point(0.0, 2.0),
        ];
        let boundary = [
            [point(0.0, 0.0), point(2.0, 0.0)],
            [point(2.0, 0.0), point(2.0, 1.0)],
            [point(2.0, 1.0), point(1.0, 1.0)],
            [point(1.0, 1.0), point(1.0, 2.0)],
            [point(1.0, 2.0), point(0.0, 2.0)],
            [point(0.0, 2.0), point(0.0, 0.0)],
        ];
        let graph = graph(&[polygon], &[point(0.5, 0.5)], &boundary);
        assert!(validate_planar_subdivision_v0(&graph, 1.0e-9, 1.0e-9).is_err());
    }
}
