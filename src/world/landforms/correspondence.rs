//! Deterministic O0b planar correspondence kernels.
//!
//! This first slice contains the preregistered geometry and mechanical best-
//! graph operations. Packet support extraction, context partitioning and
//! report-only topology are deliberately layered above these kernels.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use glam::DVec3;
use serde::{Deserialize, Serialize};

pub const O0B_CORRESPONDENCE_SCHEMA_VERSION: &str = "landform-correspondence-o0b-v0";
pub const O0B_CORRESPONDENCE_HASH_VERSION: &str = "fnv1a64-bincode-fixint-le-v0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PacketSideV0 {
    Source,
    Target,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ObjectFamilyV0 {
    Highland,
    DrainageNode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AreaSupportV0 {
    Nested,
    Exclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AreaAssignmentPolicyV0 {
    ExclusiveIntersectionAreaV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineSupportPolicyV0 {
    LocalCoveringRadiusSumV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaximumPolicyV0 {
    ExactMaximumSetV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyPolicyV0 {
    ReportOnlyTopologyV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrespondenceConfigV0 {
    pub area_policy: AreaAssignmentPolicyV0,
    pub line_policy: LineSupportPolicyV0,
    pub maximum_policy: MaximumPolicyV0,
    pub topology_policy: TopologyPolicyV0,
    pub schema_version: &'static str,
    pub hash_version: &'static str,
}

impl Default for CorrespondenceConfigV0 {
    fn default() -> Self {
        Self {
            area_policy: AreaAssignmentPolicyV0::ExclusiveIntersectionAreaV0,
            line_policy: LineSupportPolicyV0::LocalCoveringRadiusSumV0,
            maximum_policy: MaximumPolicyV0::ExactMaximumSetV0,
            topology_policy: TopologyPolicyV0::ReportOnlyTopologyV0,
            schema_version: O0B_CORRESPONDENCE_SCHEMA_VERSION,
            hash_version: O0B_CORRESPONDENCE_HASH_VERSION,
        }
    }
}

impl CorrespondenceConfigV0 {
    pub fn validate(&self) -> Result<(), CorrespondenceErrorV0> {
        if *self == Self::default() {
            Ok(())
        } else {
            Err(CorrespondenceErrorV0::UnregisteredConfiguration)
        }
    }
}

impl From<&CorrespondenceConfigV0> for CorrespondenceConfigWireV0 {
    fn from(value: &CorrespondenceConfigV0) -> Self {
        Self {
            area_policy: value.area_policy,
            line_policy: value.line_policy,
            maximum_policy: value.maximum_policy,
            topology_policy: value.topology_policy,
            schema_version: value.schema_version.into(),
            hash_version: value.hash_version.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AssignmentChannelV0 {
    HighlandExclusiveArea,
    DrainageExclusiveArea,
    DrainageLine,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentKindV0 {
    OneToOneBest,
    OneToManyBest,
    ManyToOneBest,
    ManyToManyBest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportStatusV0 {
    Eligible,
    NoExclusiveSupport,
    HierarchyAmbiguousSupport,
    NoPositiveOverlap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappedAdjacencyV0 {
    All,
    Some,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyAvailabilityV0 {
    Available,
    HierarchyAmbiguous,
    NoMappedEndpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyTargetV0 {
    Highland(u32),
    DrainageNode(u32),
    Portal(u32),
    HighlandRoot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainContextV0 {
    HighlandBackground,
    IneligibleHighlandSupport,
    Portal,
    OutsideSourceDomain,
    OutsideTargetDomain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AreaPairV0 {
    pub source_id: u32,
    pub target_id: u32,
    pub support_kind: AreaSupportV0,
    pub intersection_area_km2: f64,
    pub source_area_km2: f64,
    pub target_area_km2: f64,
    pub union_area_km2: f64,
    pub source_coverage: f64,
    pub target_coverage: f64,
    pub jaccard: f64,
    pub dice: f64,
    pub source_centroid_km: DVec3,
    pub target_centroid_km: DVec3,
    pub centroid_displacement_km: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolygonIntersectionV0 {
    pub vertices_km: Vec<DVec3>,
    pub area_km2: f64,
    pub centroid_km: DVec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LineSegmentInputV0 {
    pub endpoints_km: [DVec3; 2],
    pub measure_length_km: f64,
    pub local_radius_km: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinePairV0 {
    pub source_id: u32,
    pub target_id: u32,
    pub source_covered_length_km: f64,
    pub target_covered_length_km: f64,
    pub source_coverage: f64,
    pub target_coverage: f64,
    pub source_length_km: f64,
    pub target_length_km: f64,
    pub source_anchor_km: DVec3,
    pub target_anchor_km: DVec3,
    pub anchor_displacement_km: f64,
    pub minimum_positive_candidate_separation_km: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct LineObjectInputV0<'a> {
    pub object_id: u32,
    pub segments: &'a [LineSegmentInputV0],
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinePopulationOutputV0 {
    pub pairs: Vec<LinePairV0>,
    pub source_segments: u64,
    pub target_segments: u64,
    pub segment_box_candidates: u64,
    pub segment_pair_tests: u64,
    /// Diagnostic proof that interval queries prune separated prefixes. This
    /// is not part of the serialized O0b work-count wire.
    pub segment_index_node_visits: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineObjectMeasureV0 {
    pub total_length_km: f64,
    pub half_arclength_anchor_km: DVec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AssignmentObjectInputV0 {
    pub object_id: u32,
    pub object_measure: f64,
    pub anchor_km: DVec3,
    pub support_status: SupportStatusV0,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PositiveScoreV0 {
    pub source_id: u32,
    pub target_id: u32,
    pub source_score: f64,
    pub target_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssignmentV0 {
    pub side: PacketSideV0,
    pub family: ObjectFamilyV0,
    pub object_id: u32,
    pub channel: AssignmentChannelV0,
    pub support_status: SupportStatusV0,
    pub positive_partner_ids: Vec<u32>,
    pub maximum_partner_ids: Vec<u32>,
    pub best_score: Option<f64>,
    pub second_distinct_score: Option<f64>,
    pub normalized_margin: Option<f64>,
    pub exact_best_tie: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BestMemberV0 {
    pub side: PacketSideV0,
    pub family: ObjectFamilyV0,
    pub object_id: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BestComponentV0 {
    pub channel: AssignmentChannelV0,
    pub kind: ComponentKindV0,
    pub members: Vec<BestMemberV0>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricConflictV0 {
    pub side: PacketSideV0,
    pub drainage_node_id: u32,
    pub area_maximum_ids: Vec<u32>,
    pub line_maximum_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IneligibleHighlandAreaV0 {
    pub peak_id: u32,
    pub support_status: SupportStatusV0,
    pub area_km2: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortalAreaV0 {
    pub portal_id: u32,
    pub area_km2: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextV0 {
    pub side: PacketSideV0,
    pub family: ObjectFamilyV0,
    pub object_id: u32,
    pub background_area_km2: f64,
    pub ineligible_highland_areas: Vec<IneligibleHighlandAreaV0>,
    pub portal_areas_km2: Vec<PortalAreaV0>,
    pub outside_domain_area_km2: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyV0 {
    pub side: PacketSideV0,
    pub family: ObjectFamilyV0,
    pub channel: AssignmentChannelV0,
    pub from_id: u32,
    pub target: TopologyTargetV0,
    pub availability: TopologyAvailabilityV0,
    pub mapped_adjacency: Option<MappedAdjacencyV0>,
    pub endpoints_in_same_best_component: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyEdgeInputV0 {
    pub from_id: u32,
    pub target: TopologyTargetV0,
    pub hierarchy_ambiguous: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyObjectInputV0 {
    pub object_id: u32,
    pub target: TopologyTargetV0,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrespondenceWorkCountsV0 {
    pub source_cells: u64,
    pub target_cells: u64,
    pub cell_box_candidates: u64,
    pub polygon_clips: u64,
    pub positive_cell_intersections: u64,
    pub nested_membership_contributions: u64,
    pub source_segments: u64,
    pub target_segments: u64,
    pub segment_box_candidates: u64,
    pub segment_pair_tests: u64,
    pub positive_highland_nested_rows: u64,
    pub positive_highland_exclusive_rows: u64,
    pub positive_drainage_nested_rows: u64,
    pub positive_drainage_exclusive_rows: u64,
    pub positive_line_rows: u64,
    pub best_graph_edges: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrespondenceConfigWireV0 {
    pub area_policy: AreaAssignmentPolicyV0,
    pub line_policy: LineSupportPolicyV0,
    pub maximum_policy: MaximumPolicyV0,
    pub topology_policy: TopologyPolicyV0,
    pub schema_version: String,
    pub hash_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectCorrespondenceV0 {
    pub schema_version: String,
    pub hash_version: String,
    pub config: CorrespondenceConfigWireV0,
    pub source_packet_hash: u64,
    pub target_packet_hash: u64,
    pub highland_nested_pairs: Vec<AreaPairV0>,
    pub highland_exclusive_pairs: Vec<AreaPairV0>,
    pub drainage_nested_pairs: Vec<AreaPairV0>,
    pub drainage_exclusive_pairs: Vec<AreaPairV0>,
    pub drainage_line_pairs: Vec<LinePairV0>,
    pub context_records: Vec<ContextV0>,
    pub assignment_records: Vec<AssignmentV0>,
    pub best_components: Vec<BestComponentV0>,
    pub metric_conflicts: Vec<MetricConflictV0>,
    pub topology_records: Vec<TopologyV0>,
    pub work_counts: CorrespondenceWorkCountsV0,
    pub derived_correspondence_hash: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentKernelOutputV0 {
    pub assignments: Vec<AssignmentV0>,
    pub best_components: Vec<BestComponentV0>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CorrespondenceErrorV0 {
    UnregisteredConfiguration,
    InvalidPolygon,
    NonFiniteGeometry,
    DegenerateSegment,
    InvalidRadiusOrMeasure,
    NumericalFailure(&'static str),
    AreaBoundFailure,
    OneSidedLineCoverage,
    DuplicateObject {
        side: PacketSideV0,
        id: u32,
    },
    UnknownObject {
        side: PacketSideV0,
        id: u32,
    },
    DuplicatePositivePair {
        source_id: u32,
        target_id: u32,
    },
    DuplicateAssignment {
        side: PacketSideV0,
        id: u32,
        channel: AssignmentChannelV0,
    },
    InvalidPositiveScore,
    InvalidObjectMeasure {
        side: PacketSideV0,
        id: u32,
    },
    DuplicateTopologyObject(u32),
    DuplicateTopologyEdge(u32),
    MissingTopologyObject(u32),
    InvalidTopologyTarget(TopologyTargetV0),
    UndeclaredPortal(u32),
}

impl fmt::Display for CorrespondenceErrorV0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CorrespondenceErrorV0 {}

pub fn convex_polygon_intersection_v0(
    a: &[DVec3],
    b: &[DVec3],
    endpoint_tolerance_km: f64,
) -> Result<Option<PolygonIntersectionV0>, CorrespondenceErrorV0> {
    if !endpoint_tolerance_km.is_finite() || endpoint_tolerance_km < 0.0 {
        return Err(CorrespondenceErrorV0::NonFiniteGeometry);
    }
    let a = canonical_polygon(a, endpoint_tolerance_km)?;
    let b = canonical_polygon(b, endpoint_tolerance_km)?;
    let (subject, clip) = if polygon_sequence_cmp(&a, &b) != Ordering::Greater {
        (&a, &b)
    } else {
        (&b, &a)
    };
    let mut output = subject.clone();
    for edge in 0..clip.len() {
        let edge_start = clip[edge];
        let edge_end = clip[(edge + 1) % clip.len()];
        let edge_vector = edge_end - edge_start;
        let boundary_tolerance = endpoint_tolerance_km * edge_vector.length();
        let input = std::mem::take(&mut output);
        if input.is_empty() {
            return Ok(None);
        }
        let mut previous = *input.last().expect("nonempty clip input");
        let mut previous_cross = cross2(edge_vector, previous - edge_start);
        let mut previous_inside = previous_cross >= -boundary_tolerance;
        for current in input {
            let current_cross = cross2(edge_vector, current - edge_start);
            let current_inside = current_cross >= -boundary_tolerance;
            if current_inside != previous_inside {
                let denominator = previous_cross - current_cross;
                if denominator == 0.0 || !denominator.is_finite() {
                    return Err(CorrespondenceErrorV0::NumericalFailure("polygon crossing"));
                }
                let t = previous_cross / denominator;
                let intersection = previous + t * (current - previous);
                if !intersection.is_finite() {
                    return Err(CorrespondenceErrorV0::NonFiniteGeometry);
                }
                output.push(DVec3::new(intersection.x, intersection.y, 0.0));
            }
            if current_inside {
                output.push(current);
            }
            previous = current;
            previous_cross = current_cross;
            previous_inside = current_inside;
        }
        output = weld_polygon(output, endpoint_tolerance_km);
    }
    output = weld_polygon(output, endpoint_tolerance_km);
    if output.len() < 3 {
        return Ok(None);
    }
    rotate_polygon(&mut output);
    let (area, centroid) = polygon_area_centroid(&output)?;
    if area <= 0.0 {
        return Ok(None);
    }
    Ok(Some(PolygonIntersectionV0 {
        vertices_km: output,
        area_km2: area,
        centroid_km: centroid,
    }))
}

pub fn build_area_pair_v0(
    source_id: u32,
    target_id: u32,
    support_kind: AreaSupportV0,
    source_polygons: &[Vec<DVec3>],
    target_polygons: &[Vec<DVec3>],
    endpoint_tolerance_km: f64,
    planar_area_match_relative: f64,
) -> Result<Option<AreaPairV0>, CorrespondenceErrorV0> {
    if !planar_area_match_relative.is_finite() || planar_area_match_relative < 0.0 {
        return Err(CorrespondenceErrorV0::NonFiniteGeometry);
    }
    let source = prepare_support(source_polygons, endpoint_tolerance_km)?;
    let target = prepare_support(target_polygons, endpoint_tolerance_km)?;
    if source.total_area <= 0.0 || target.total_area <= 0.0 {
        return Err(CorrespondenceErrorV0::InvalidPolygon);
    }

    let mut target_by_min_x = (0..target.polygons.len()).collect::<Vec<_>>();
    target_by_min_x.sort_by(|&i, &j| {
        target.boxes[i]
            .min_x
            .total_cmp(&target.boxes[j].min_x)
            .then_with(|| polygon_sequence_cmp(&target.polygons[i], &target.polygons[j]))
    });
    let mut contributions = Vec::new();
    for (source_index, source_polygon) in source.polygons.iter().enumerate() {
        let source_box = source.boxes[source_index];
        let end = target_by_min_x
            .partition_point(|&target_index| target.boxes[target_index].min_x <= source_box.max_x);
        for &target_index in &target_by_min_x[..end] {
            let target_box = target.boxes[target_index];
            if !source_box.overlaps(target_box) {
                continue;
            }
            if let Some(intersection) = convex_polygon_intersection_v0(
                source_polygon,
                &target.polygons[target_index],
                endpoint_tolerance_km,
            )? {
                let first = source_polygon;
                let second = &target.polygons[target_index];
                let (low, high) = if polygon_sequence_cmp(first, second) != Ordering::Greater {
                    (first.clone(), second.clone())
                } else {
                    (second.clone(), first.clone())
                };
                contributions.push((
                    low,
                    high,
                    intersection.area_km2,
                    intersection.area_km2 * intersection.centroid_km,
                ));
            }
        }
    }
    contributions.sort_by(|a, b| {
        polygon_sequence_cmp(&a.0, &b.0).then_with(|| polygon_sequence_cmp(&a.1, &b.1))
    });
    let intersection_area = kahan_sum(contributions.iter().map(|value| value.2))?;
    if intersection_area == 0.0 {
        return Ok(None);
    }
    let tolerance = endpoint_tolerance_km * endpoint_tolerance_km
        + planar_area_match_relative * source.total_area.max(target.total_area);
    let maximum = source.total_area.min(target.total_area);
    let intersection_area = if intersection_area > maximum {
        if intersection_area - maximum > tolerance {
            return Err(CorrespondenceErrorV0::AreaBoundFailure);
        }
        maximum
    } else {
        intersection_area
    };
    let source_coverage = intersection_area / source.total_area;
    let target_coverage = intersection_area / target.total_area;
    if source_coverage > 1.0 || target_coverage > 1.0 {
        return Err(CorrespondenceErrorV0::AreaBoundFailure);
    }
    let union = source.total_area + target.total_area - intersection_area;
    Ok(Some(AreaPairV0 {
        source_id,
        target_id,
        support_kind,
        intersection_area_km2: intersection_area,
        source_area_km2: source.total_area,
        target_area_km2: target.total_area,
        union_area_km2: union,
        source_coverage,
        target_coverage,
        jaccard: intersection_area / union,
        dice: 2.0 * intersection_area / (source.total_area + target.total_area),
        source_centroid_km: source.centroid,
        target_centroid_km: target.centroid,
        centroid_displacement_km: source.centroid.distance(target.centroid),
    }))
}

#[derive(Clone, Copy)]
struct BoundingBox {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl BoundingBox {
    fn overlaps(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && other.min_x <= self.max_x
            && self.min_y <= other.max_y
            && other.min_y <= self.max_y
    }
}

struct PreparedSupport {
    polygons: Vec<Vec<DVec3>>,
    boxes: Vec<BoundingBox>,
    total_area: f64,
    centroid: DVec3,
}

fn prepare_support(
    polygons: &[Vec<DVec3>],
    tolerance: f64,
) -> Result<PreparedSupport, CorrespondenceErrorV0> {
    let mut polygons = polygons
        .iter()
        .map(|polygon| canonical_polygon(polygon, tolerance))
        .collect::<Result<Vec<_>, _>>()?;
    polygons.sort_by(|a, b| polygon_sequence_cmp(a, b));
    let mut boxes = Vec::with_capacity(polygons.len());
    let mut areas = Vec::with_capacity(polygons.len());
    let mut moments = Vec::with_capacity(polygons.len());
    for polygon in &polygons {
        let (area, centroid) = polygon_area_centroid(polygon)?;
        areas.push(area);
        moments.push(area * centroid);
        boxes.push(polygon_box(polygon));
    }
    let total_area = kahan_sum(areas.iter().copied())?;
    if total_area <= 0.0 {
        return Err(CorrespondenceErrorV0::InvalidPolygon);
    }
    let centroid = kahan_vec_sum(moments.iter().copied())? / total_area;
    Ok(PreparedSupport {
        polygons,
        boxes,
        total_area,
        centroid,
    })
}

fn canonical_polygon(
    polygon: &[DVec3],
    tolerance: f64,
) -> Result<Vec<DVec3>, CorrespondenceErrorV0> {
    if polygon.iter().any(|point| !point.is_finite()) {
        return Err(CorrespondenceErrorV0::NonFiniteGeometry);
    }
    if polygon.len() < 3 || polygon.iter().any(|point| point.z != 0.0) {
        return Err(CorrespondenceErrorV0::InvalidPolygon);
    }
    let mut polygon = weld_polygon(polygon.to_vec(), tolerance);
    if polygon.len() < 3 {
        return Err(CorrespondenceErrorV0::InvalidPolygon);
    }
    rotate_polygon(&mut polygon);
    let (area, _) = polygon_area_centroid(&polygon)?;
    if area <= 0.0 || !is_convex_ccw(&polygon, tolerance) {
        return Err(CorrespondenceErrorV0::InvalidPolygon);
    }
    Ok(polygon)
}

fn weld_polygon(mut polygon: Vec<DVec3>, tolerance: f64) -> Vec<DVec3> {
    let mut welded = Vec::with_capacity(polygon.len());
    for mut point in polygon.drain(..) {
        point.z = 0.0;
        point.x = canonical_zero(point.x);
        point.y = canonical_zero(point.y);
        if welded
            .last()
            .is_none_or(|previous: &DVec3| previous.distance(point) > tolerance)
        {
            welded.push(point);
        }
    }
    if welded.len() > 1 && welded[0].distance(*welded.last().unwrap()) <= tolerance {
        welded.pop();
    }
    welded
}

fn rotate_polygon(polygon: &mut [DVec3]) {
    if let Some((start, _)) = polygon
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| point_cmp(**a, **b))
    {
        polygon.rotate_left(start);
    }
}

fn polygon_sequence_cmp(a: &[DVec3], b: &[DVec3]) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| {
        a.iter()
            .zip(b)
            .map(|(&a, &b)| point_cmp(a, b))
            .find(|ordering| *ordering != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    })
}

fn point_cmp(a: DVec3, b: DVec3) -> Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.z.total_cmp(&b.z))
}

fn polygon_box(polygon: &[DVec3]) -> BoundingBox {
    let mut result = BoundingBox {
        min_x: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        min_y: f64::INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    for point in polygon {
        result.min_x = result.min_x.min(point.x);
        result.max_x = result.max_x.max(point.x);
        result.min_y = result.min_y.min(point.y);
        result.max_y = result.max_y.max(point.y);
    }
    result
}

fn is_convex_ccw(polygon: &[DVec3], tolerance: f64) -> bool {
    (0..polygon.len()).all(|index| {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        let c = polygon[(index + 2) % polygon.len()];
        cross2(b - a, c - b) >= -tolerance * (b - a).length()
    })
}

fn polygon_area_centroid(polygon: &[DVec3]) -> Result<(f64, DVec3), CorrespondenceErrorV0> {
    let mut cross_values = Vec::with_capacity(polygon.len());
    let mut x_values = Vec::with_capacity(polygon.len());
    let mut y_values = Vec::with_capacity(polygon.len());
    for index in 0..polygon.len() {
        let a = polygon[index];
        let b = polygon[(index + 1) % polygon.len()];
        let cross = a.x * b.y - b.x * a.y;
        cross_values.push(cross);
        x_values.push((a.x + b.x) * cross);
        y_values.push((a.y + b.y) * cross);
    }
    let twice_area = kahan_sum(cross_values)?;
    if !twice_area.is_finite() || twice_area <= 0.0 {
        return Err(CorrespondenceErrorV0::InvalidPolygon);
    }
    let centroid = DVec3::new(
        kahan_sum(x_values)? / (3.0 * twice_area),
        kahan_sum(y_values)? / (3.0 * twice_area),
        0.0,
    );
    if !centroid.is_finite() {
        return Err(CorrespondenceErrorV0::NumericalFailure("polygon centroid"));
    }
    Ok((0.5 * twice_area, centroid))
}

fn cross2(a: DVec3, b: DVec3) -> f64 {
    a.x * b.y - a.y * b.x
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn kahan_sum(values: impl IntoIterator<Item = f64>) -> Result<f64, CorrespondenceErrorV0> {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        if !sum.is_finite() || !correction.is_finite() {
            return Err(CorrespondenceErrorV0::NumericalFailure("compensated sum"));
        }
    }
    Ok(if sum == 0.0 { 0.0 } else { sum })
}

fn kahan_vec_sum(values: impl IntoIterator<Item = DVec3>) -> Result<DVec3, CorrespondenceErrorV0> {
    let values = values.into_iter().collect::<Vec<_>>();
    Ok(DVec3::new(
        kahan_sum(values.iter().map(|value| value.x))?,
        kahan_sum(values.iter().map(|value| value.y))?,
        kahan_sum(values.iter().map(|value| value.z))?,
    ))
}

pub fn build_line_pair_v0(
    source_id: u32,
    target_id: u32,
    source_segments: &[LineSegmentInputV0],
    target_segments: &[LineSegmentInputV0],
) -> Result<Option<LinePairV0>, CorrespondenceErrorV0> {
    let output = build_line_population_v0(
        &[LineObjectInputV0 {
            object_id: source_id,
            segments: source_segments,
        }],
        &[LineObjectInputV0 {
            object_id: target_id,
            segments: target_segments,
        }],
    )?;
    Ok(output.pairs.into_iter().next())
}

pub fn measure_line_object_v0(
    segments: &[LineSegmentInputV0],
) -> Result<LineObjectMeasureV0, CorrespondenceErrorV0> {
    validate_segments(segments)?;
    let total_length_km = kahan_sum(segments.iter().map(|segment| segment.measure_length_km))?;
    Ok(LineObjectMeasureV0 {
        total_length_km,
        half_arclength_anchor_km: line_anchor(segments, total_length_km)?,
    })
}

pub fn build_line_population_v0(
    source_lines: &[LineObjectInputV0<'_>],
    target_lines: &[LineObjectInputV0<'_>],
) -> Result<LinePopulationOutputV0, CorrespondenceErrorV0> {
    struct PreparedLine<'a> {
        id: u32,
        segments: &'a [LineSegmentInputV0],
        total_length: f64,
        anchor: DVec3,
    }
    struct PairState {
        source_intervals: Vec<Vec<[f64; 2]>>,
        target_intervals: Vec<Vec<[f64; 2]>>,
        minimum_separation: f64,
    }

    fn prepare_lines<'a>(
        side: PacketSideV0,
        lines: &[LineObjectInputV0<'a>],
    ) -> Result<Vec<PreparedLine<'a>>, CorrespondenceErrorV0> {
        let mut prepared = Vec::with_capacity(lines.len());
        let mut ids = BTreeSet::new();
        for line in lines {
            if !ids.insert(line.object_id) {
                return Err(CorrespondenceErrorV0::DuplicateObject {
                    side,
                    id: line.object_id,
                });
            }
            let measure = measure_line_object_v0(line.segments)?;
            prepared.push(PreparedLine {
                id: line.object_id,
                segments: line.segments,
                total_length: measure.total_length_km,
                anchor: measure.half_arclength_anchor_km,
            });
        }
        prepared.sort_by_key(|line| line.id);
        Ok(prepared)
    }

    let source = prepare_lines(PacketSideV0::Source, source_lines)?;
    let target = prepare_lines(PacketSideV0::Target, target_lines)?;
    let source_segments = source.iter().try_fold(0u64, |count, line| {
        count.checked_add(line.segments.len() as u64).ok_or(
            CorrespondenceErrorV0::NumericalFailure("source segment count overflow"),
        )
    })?;
    let target_segments = target.iter().try_fold(0u64, |count, line| {
        count.checked_add(line.segments.len() as u64).ok_or(
            CorrespondenceErrorV0::NumericalFailure("target segment count overflow"),
        )
    })?;
    let mut target_segment_records = Vec::with_capacity(target_segments as usize);
    for (line_index, line) in target.iter().enumerate() {
        for (segment_index, &segment) in line.segments.iter().enumerate() {
            target_segment_records.push(IndexedLineTargetSegment {
                line_index,
                segment_index,
                segment,
                bounds: segment_box(segment),
                key: segment_key(segment),
            });
        }
    }
    let target_index = SegmentIntervalIndex::new(&target_segment_records);

    let mut states: BTreeMap<(usize, usize), PairState> = BTreeMap::new();
    let mut segment_box_candidates = 0u64;
    let mut segment_pair_tests = 0u64;
    let mut segment_index_node_visits = 0u64;
    for (source_line_index, source_line) in source.iter().enumerate() {
        for (source_segment_index, &source_segment) in source_line.segments.iter().enumerate() {
            let source_bounds = segment_box(source_segment);
            for candidate_index in target_index.query(
                &target_segment_records,
                source_bounds,
                &mut segment_index_node_visits,
            )? {
                let candidate = &target_segment_records[candidate_index];
                segment_box_candidates = segment_box_candidates.checked_add(1).ok_or(
                    CorrespondenceErrorV0::NumericalFailure("segment candidate count overflow"),
                )?;
                segment_pair_tests = segment_pair_tests.checked_add(1).ok_or(
                    CorrespondenceErrorV0::NumericalFailure("segment test count overflow"),
                )?;
                let solution = solve_segment_pair(source_segment, candidate.segment)?;
                let state = states
                    .entry((source_line_index, candidate.line_index))
                    .or_insert_with(|| PairState {
                        source_intervals: vec![Vec::new(); source_line.segments.len()],
                        target_intervals: vec![
                            Vec::new();
                            target[candidate.line_index].segments.len()
                        ],
                        minimum_separation: f64::INFINITY,
                    });
                state.source_intervals[source_segment_index].extend(solution.source_intervals);
                state.target_intervals[candidate.segment_index].extend(solution.target_intervals);
                state.minimum_separation = state.minimum_separation.min(solution.separation_km);
            }
        }
    }

    let mut pairs = Vec::new();
    for ((source_index, target_index), mut state) in states {
        let source_line = &source[source_index];
        let target_line = &target[target_index];
        let source_covered = covered_measure(source_line.segments, &mut state.source_intervals)?;
        let target_covered = covered_measure(target_line.segments, &mut state.target_intervals)?;
        if (source_covered > 0.0) != (target_covered > 0.0) {
            return Err(CorrespondenceErrorV0::OneSidedLineCoverage);
        }
        if source_covered == 0.0 {
            continue;
        }
        if !state.minimum_separation.is_finite() {
            return Err(CorrespondenceErrorV0::NumericalFailure(
                "candidate separation",
            ));
        }
        pairs.push(LinePairV0 {
            source_id: source_line.id,
            target_id: target_line.id,
            source_covered_length_km: source_covered,
            target_covered_length_km: target_covered,
            source_coverage: source_covered / source_line.total_length,
            target_coverage: target_covered / target_line.total_length,
            source_length_km: source_line.total_length,
            target_length_km: target_line.total_length,
            source_anchor_km: source_line.anchor,
            target_anchor_km: target_line.anchor,
            anchor_displacement_km: source_line.anchor.distance(target_line.anchor),
            minimum_positive_candidate_separation_km: state.minimum_separation,
        });
    }
    Ok(LinePopulationOutputV0 {
        pairs,
        source_segments,
        target_segments,
        segment_box_candidates,
        segment_pair_tests,
        segment_index_node_visits,
    })
}

#[derive(Clone)]
struct SegmentKey {
    endpoints: [DVec3; 2],
    measure: f64,
    radius: f64,
}

struct IndexedLineTargetSegment {
    line_index: usize,
    segment_index: usize,
    segment: LineSegmentInputV0,
    bounds: BoundingBox,
    key: SegmentKey,
}

struct SegmentIntervalIndex {
    order: Vec<usize>,
    leaf_count: usize,
    subtree_max_x: Vec<f64>,
}

impl SegmentIntervalIndex {
    fn new(segments: &[IndexedLineTargetSegment]) -> Self {
        let mut order = (0..segments.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            segments[a]
                .bounds
                .min_x
                .total_cmp(&segments[b].bounds.min_x)
                .then_with(|| segment_key_cmp(&segments[a].key, &segments[b].key))
                .then_with(|| segments[a].line_index.cmp(&segments[b].line_index))
                .then_with(|| segments[a].segment_index.cmp(&segments[b].segment_index))
        });
        let leaf_count = order.len().next_power_of_two().max(1);
        let mut subtree_max_x = vec![f64::NEG_INFINITY; 2 * leaf_count];
        for (offset, &segment) in order.iter().enumerate() {
            subtree_max_x[leaf_count + offset] = segments[segment].bounds.max_x;
        }
        for node in (1..leaf_count).rev() {
            subtree_max_x[node] = subtree_max_x[2 * node].max(subtree_max_x[2 * node + 1]);
        }
        Self {
            order,
            leaf_count,
            subtree_max_x,
        }
    }

    fn query(
        &self,
        segments: &[IndexedLineTargetSegment],
        bounds: BoundingBox,
        node_visits: &mut u64,
    ) -> Result<Vec<usize>, CorrespondenceErrorV0> {
        let query_end = self
            .order
            .partition_point(|&segment| segments[segment].bounds.min_x <= bounds.max_x);
        let mut result = Vec::new();
        self.query_node(
            segments,
            bounds,
            1,
            0,
            self.leaf_count,
            query_end,
            node_visits,
            &mut result,
        )?;
        result.sort_by(|&a, &b| {
            segment_key_cmp(&segments[a].key, &segments[b].key)
                .then_with(|| segments[a].line_index.cmp(&segments[b].line_index))
                .then_with(|| segments[a].segment_index.cmp(&segments[b].segment_index))
        });
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn query_node(
        &self,
        segments: &[IndexedLineTargetSegment],
        bounds: BoundingBox,
        node: usize,
        begin: usize,
        end: usize,
        query_end: usize,
        node_visits: &mut u64,
        result: &mut Vec<usize>,
    ) -> Result<(), CorrespondenceErrorV0> {
        *node_visits =
            node_visits
                .checked_add(1)
                .ok_or(CorrespondenceErrorV0::NumericalFailure(
                    "segment index visit count overflow",
                ))?;
        if begin >= query_end || self.subtree_max_x[node] < bounds.min_x {
            return Ok(());
        }
        if end - begin == 1 {
            if begin < self.order.len() {
                let segment = self.order[begin];
                if bounds.overlaps(segments[segment].bounds) {
                    result.push(segment);
                }
            }
            return Ok(());
        }
        let middle = (begin + end) / 2;
        self.query_node(
            segments,
            bounds,
            2 * node,
            begin,
            middle,
            query_end,
            node_visits,
            result,
        )?;
        self.query_node(
            segments,
            bounds,
            2 * node + 1,
            middle,
            end,
            query_end,
            node_visits,
            result,
        )
    }
}

fn segment_key(segment: LineSegmentInputV0) -> SegmentKey {
    let mut endpoints = segment.endpoints_km;
    if point_cmp(endpoints[0], endpoints[1]) == Ordering::Greater {
        endpoints.swap(0, 1);
    }
    SegmentKey {
        endpoints,
        measure: segment.measure_length_km,
        radius: segment.local_radius_km,
    }
}

fn segment_key_cmp(a: &SegmentKey, b: &SegmentKey) -> Ordering {
    point_cmp(a.endpoints[0], b.endpoints[0])
        .then_with(|| point_cmp(a.endpoints[1], b.endpoints[1]))
        .then_with(|| a.measure.total_cmp(&b.measure))
        .then_with(|| a.radius.total_cmp(&b.radius))
}

struct SegmentPairSolution {
    source_intervals: Vec<[f64; 2]>,
    target_intervals: Vec<[f64; 2]>,
    separation_km: f64,
}

fn solve_segment_pair(
    source: LineSegmentInputV0,
    target: LineSegmentInputV0,
) -> Result<SegmentPairSolution, CorrespondenceErrorV0> {
    let source_key = segment_key(source);
    let target_key = segment_key(target);
    let radius = source.local_radius_km + target.local_radius_km;
    if segment_key_cmp(&source_key, &target_key) != Ordering::Greater {
        Ok(SegmentPairSolution {
            source_intervals: covered_parameter_intervals(source, target, radius)?,
            target_intervals: covered_parameter_intervals(target, source, radius)?,
            separation_km: segment_separation(source, target)?,
        })
    } else {
        let target_intervals = covered_parameter_intervals(target, source, radius)?;
        let source_intervals = covered_parameter_intervals(source, target, radius)?;
        Ok(SegmentPairSolution {
            source_intervals,
            target_intervals,
            separation_km: segment_separation(target, source)?,
        })
    }
}

fn validate_segments(segments: &[LineSegmentInputV0]) -> Result<(), CorrespondenceErrorV0> {
    if segments.is_empty() {
        return Err(CorrespondenceErrorV0::DegenerateSegment);
    }
    for segment in segments {
        if segment
            .endpoints_km
            .iter()
            .any(|point| !point.is_finite() || point.z != 0.0)
        {
            return Err(CorrespondenceErrorV0::NonFiniteGeometry);
        }
        if segment.endpoints_km[0] == segment.endpoints_km[1] {
            return Err(CorrespondenceErrorV0::DegenerateSegment);
        }
        if !segment.measure_length_km.is_finite()
            || segment.measure_length_km <= 0.0
            || !segment.local_radius_km.is_finite()
            || segment.local_radius_km < 0.0
        {
            return Err(CorrespondenceErrorV0::InvalidRadiusOrMeasure);
        }
    }
    Ok(())
}

fn segment_box(segment: LineSegmentInputV0) -> BoundingBox {
    let radius = segment.local_radius_km;
    BoundingBox {
        min_x: segment.endpoints_km[0].x.min(segment.endpoints_km[1].x) - radius,
        max_x: segment.endpoints_km[0].x.max(segment.endpoints_km[1].x) + radius,
        min_y: segment.endpoints_km[0].y.min(segment.endpoints_km[1].y) - radius,
        max_y: segment.endpoints_km[0].y.max(segment.endpoints_km[1].y) + radius,
    }
}

fn covered_parameter_intervals(
    source: LineSegmentInputV0,
    target: LineSegmentInputV0,
    radius: f64,
) -> Result<Vec<[f64; 2]>, CorrespondenceErrorV0> {
    let a = source.endpoints_km[0];
    let u = source.endpoints_km[1] - a;
    let c = target.endpoints_km[0];
    let v = target.endpoints_km[1] - c;
    let vv = v.length_squared();
    if vv <= 0.0 || !vv.is_finite() || !radius.is_finite() {
        return Err(CorrespondenceErrorV0::DegenerateSegment);
    }
    let projection_start = (a - c).dot(v) / vv;
    let projection_slope = u.dot(v) / vv;
    let mut splits = vec![0.0, 1.0];
    if projection_slope != 0.0 {
        for boundary in [0.0, 1.0] {
            let split = (boundary - projection_start) / projection_slope;
            if split > 0.0 && split < 1.0 && split.is_finite() {
                splits.push(split);
            }
        }
    }
    splits.sort_by(f64::total_cmp);
    splits.dedup_by(|a, b| a.total_cmp(b) == Ordering::Equal);
    let radius_squared = radius * radius;
    let mut result = Vec::new();
    for window in splits.windows(2) {
        let lo = window[0];
        let hi = window[1];
        if hi <= lo {
            continue;
        }
        let mid = 0.5 * (lo + hi);
        let projected = projection_start + projection_slope * mid;
        let (quadratic_a, quadratic_b, quadratic_c) = if projected <= 0.0 {
            point_distance_quadratic(a - c, u)
        } else if projected >= 1.0 {
            point_distance_quadratic(a - target.endpoints_km[1], u)
        } else {
            let w = a - c;
            let cross_start = cross2(v, w);
            let cross_slope = cross2(v, u);
            let a_coefficient = cross_slope * cross_slope / vv;
            let b_coefficient = 2.0 * cross_start * cross_slope / vv;
            let c_coefficient = cross_start * cross_start / vv;
            (a_coefficient, b_coefficient, c_coefficient)
        };
        if let Some(interval) = solve_quadratic_interval(
            quadratic_a,
            quadratic_b,
            quadratic_c,
            radius_squared,
            lo,
            hi,
        )? {
            result.push(interval);
        }
    }
    union_intervals(&mut result);
    Ok(result)
}

fn point_distance_quadratic(w: DVec3, u: DVec3) -> (f64, f64, f64) {
    (u.length_squared(), 2.0 * w.dot(u), w.length_squared())
}

fn solve_quadratic_interval(
    a: f64,
    b: f64,
    c: f64,
    radius_squared: f64,
    lo: f64,
    hi: f64,
) -> Result<Option<[f64; 2]>, CorrespondenceErrorV0> {
    let shifted_c = c - radius_squared;
    if [a, b, shifted_c, lo, hi]
        .into_iter()
        .any(|value| !value.is_finite())
    {
        return Err(CorrespondenceErrorV0::NumericalFailure(
            "nonfinite quadratic",
        ));
    }
    if a < 0.0 {
        return Err(CorrespondenceErrorV0::NumericalFailure(
            "negative quadratic coefficient",
        ));
    }
    let (pass_lo, pass_hi) = if a == 0.0 {
        if b == 0.0 {
            if shifted_c <= 0.0 {
                (lo, hi)
            } else {
                return Ok(None);
            }
        } else {
            let root = -shifted_c / b;
            if b > 0.0 {
                (lo, hi.min(root))
            } else {
                (lo.max(root), hi)
            }
        }
    } else {
        let discriminant = b * b - 4.0 * a * shifted_c;
        if !discriminant.is_finite() {
            return Err(CorrespondenceErrorV0::NumericalFailure(
                "quadratic discriminant",
            ));
        }
        if discriminant <= 0.0 {
            return Ok(None);
        }
        let q = -0.5 * (b + discriminant.sqrt().copysign(b));
        let (root_a, root_b) = if q == 0.0 {
            let root = -b / (2.0 * a);
            (root, root)
        } else {
            (q / a, shifted_c / q)
        };
        let lower = root_a.min(root_b);
        let upper = root_a.max(root_b);
        (lo.max(lower), hi.min(upper))
    };
    if pass_hi > pass_lo {
        Ok(Some([pass_lo, pass_hi]))
    } else {
        Ok(None)
    }
}

fn union_intervals(intervals: &mut Vec<[f64; 2]>) {
    intervals.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    let mut merged: Vec<[f64; 2]> = Vec::with_capacity(intervals.len());
    for interval in intervals.drain(..) {
        if let Some(last) = merged.last_mut() {
            if interval[0] <= last[1] {
                last[1] = last[1].max(interval[1]);
                continue;
            }
        }
        merged.push(interval);
    }
    *intervals = merged;
}

fn covered_measure(
    segments: &[LineSegmentInputV0],
    intervals: &mut [Vec<[f64; 2]>],
) -> Result<f64, CorrespondenceErrorV0> {
    let mut contributions = Vec::new();
    for (segment, intervals) in segments.iter().zip(intervals) {
        union_intervals(intervals);
        let measure = kahan_sum(
            intervals
                .iter()
                .map(|interval| (interval[1] - interval[0]) * segment.measure_length_km),
        )?;
        contributions.push((segment_key(*segment), measure));
    }
    contributions.sort_by(|a, b| segment_key_cmp(&a.0, &b.0));
    kahan_sum(contributions.into_iter().map(|value| value.1))
}

fn line_anchor(
    segments: &[LineSegmentInputV0],
    total_length: f64,
) -> Result<DVec3, CorrespondenceErrorV0> {
    let half = 0.5 * total_length;
    let mut preceding = 0.0;
    for segment in segments {
        let end = preceding + segment.measure_length_km;
        if half <= end {
            let fraction = (half - preceding) / segment.measure_length_km;
            let anchor = segment.endpoints_km[0]
                + fraction * (segment.endpoints_km[1] - segment.endpoints_km[0]);
            if anchor.is_finite() {
                return Ok(anchor);
            }
            return Err(CorrespondenceErrorV0::NumericalFailure("line anchor"));
        }
        preceding = end;
    }
    Err(CorrespondenceErrorV0::NumericalFailure("line anchor"))
}

fn segment_separation(
    source: LineSegmentInputV0,
    target: LineSegmentInputV0,
) -> Result<f64, CorrespondenceErrorV0> {
    let mut minimum_squared = f64::INFINITY;
    for (from, to) in [(source, target), (target, source)] {
        let a = from.endpoints_km[0];
        let u = from.endpoints_km[1] - a;
        let c = to.endpoints_km[0];
        let v = to.endpoints_km[1] - c;
        let vv = v.length_squared();
        let projection_start = (a - c).dot(v) / vv;
        let projection_slope = u.dot(v) / vv;
        let mut splits = vec![0.0, 1.0];
        if projection_slope != 0.0 {
            for boundary in [0.0, 1.0] {
                let split = (boundary - projection_start) / projection_slope;
                if split > 0.0 && split < 1.0 && split.is_finite() {
                    splits.push(split);
                }
            }
        }
        splits.sort_by(f64::total_cmp);
        splits.dedup_by(|a, b| a.total_cmp(b) == Ordering::Equal);
        for window in splits.windows(2) {
            let lo = window[0];
            let hi = window[1];
            let mid = 0.5 * (lo + hi);
            let projection = projection_start + projection_slope * mid;
            let (qa, qb, qc) = if projection <= 0.0 {
                point_distance_quadratic(a - c, u)
            } else if projection >= 1.0 {
                point_distance_quadratic(a - to.endpoints_km[1], u)
            } else {
                let w = a - c;
                let cross_start = cross2(v, w);
                let cross_slope = cross2(v, u);
                (
                    cross_slope * cross_slope / vv,
                    2.0 * cross_start * cross_slope / vv,
                    cross_start * cross_start / vv,
                )
            };
            let t = if qa > 0.0 {
                (-qb / (2.0 * qa)).clamp(lo, hi)
            } else if qb > 0.0 {
                lo
            } else {
                hi
            };
            let squared = qa * t * t + qb * t + qc;
            minimum_squared = minimum_squared.min(squared.max(0.0));
        }
    }
    let separation = minimum_squared.sqrt();
    if separation.is_finite() {
        Ok(separation)
    } else {
        Err(CorrespondenceErrorV0::NumericalFailure(
            "segment separation",
        ))
    }
}

pub fn build_assignment_kernel_v0(
    family: ObjectFamilyV0,
    channel: AssignmentChannelV0,
    source_objects: &[AssignmentObjectInputV0],
    target_objects: &[AssignmentObjectInputV0],
    positive_scores: &[PositiveScoreV0],
) -> Result<AssignmentKernelOutputV0, CorrespondenceErrorV0> {
    match (family, channel) {
        (ObjectFamilyV0::Highland, AssignmentChannelV0::HighlandExclusiveArea)
        | (ObjectFamilyV0::DrainageNode, AssignmentChannelV0::DrainageExclusiveArea)
        | (ObjectFamilyV0::DrainageNode, AssignmentChannelV0::DrainageLine) => {}
        _ => return Err(CorrespondenceErrorV0::UnregisteredConfiguration),
    }
    let source = validate_assignment_objects(PacketSideV0::Source, source_objects)?;
    let target = validate_assignment_objects(PacketSideV0::Target, target_objects)?;
    let mut source_scores: BTreeMap<u32, Vec<(u32, f64)>> = BTreeMap::new();
    let mut target_scores: BTreeMap<u32, Vec<(u32, f64)>> = BTreeMap::new();
    let mut pairs = BTreeSet::new();
    for score in positive_scores {
        if !source.contains_key(&score.source_id) {
            return Err(CorrespondenceErrorV0::UnknownObject {
                side: PacketSideV0::Source,
                id: score.source_id,
            });
        }
        if !target.contains_key(&score.target_id) {
            return Err(CorrespondenceErrorV0::UnknownObject {
                side: PacketSideV0::Target,
                id: score.target_id,
            });
        }
        if !score.source_score.is_finite()
            || score.source_score <= 0.0
            || !score.target_score.is_finite()
            || score.target_score <= 0.0
        {
            return Err(CorrespondenceErrorV0::InvalidPositiveScore);
        }
        if source[&score.source_id].support_status != SupportStatusV0::Eligible
            || target[&score.target_id].support_status != SupportStatusV0::Eligible
        {
            return Err(CorrespondenceErrorV0::InvalidPositiveScore);
        }
        if !pairs.insert((score.source_id, score.target_id)) {
            return Err(CorrespondenceErrorV0::DuplicatePositivePair {
                source_id: score.source_id,
                target_id: score.target_id,
            });
        }
        source_scores
            .entry(score.source_id)
            .or_default()
            .push((score.target_id, score.source_score));
        target_scores
            .entry(score.target_id)
            .or_default()
            .push((score.source_id, score.target_score));
    }

    let mut assignments = Vec::with_capacity(source.len() + target.len());
    for (&id, object) in &source {
        assignments.push(build_assignment(
            PacketSideV0::Source,
            family,
            channel,
            object,
            source_scores.remove(&id).unwrap_or_default(),
        )?);
    }
    for (&id, object) in &target {
        assignments.push(build_assignment(
            PacketSideV0::Target,
            family,
            channel,
            object,
            target_scores.remove(&id).unwrap_or_default(),
        )?);
    }
    let best_components = build_best_components(family, channel, &assignments, &source, &target)?;
    Ok(AssignmentKernelOutputV0 {
        assignments,
        best_components,
    })
}

pub fn build_metric_conflicts_v0(
    assignments: &[AssignmentV0],
) -> Result<Vec<MetricConflictV0>, CorrespondenceErrorV0> {
    let mut maxima: BTreeMap<(PacketSideV0, u32), [Option<Vec<u32>>; 2]> = BTreeMap::new();
    for assignment in assignments {
        if assignment.family != ObjectFamilyV0::DrainageNode {
            continue;
        }
        let channel_index = match assignment.channel {
            AssignmentChannelV0::DrainageExclusiveArea => 0,
            AssignmentChannelV0::DrainageLine => 1,
            AssignmentChannelV0::HighlandExclusiveArea => continue,
        };
        let entry = maxima
            .entry((assignment.side, assignment.object_id))
            .or_insert_with(|| [None, None]);
        if entry[channel_index]
            .replace(assignment.maximum_partner_ids.clone())
            .is_some()
        {
            return Err(CorrespondenceErrorV0::DuplicateAssignment {
                side: assignment.side,
                id: assignment.object_id,
                channel: assignment.channel,
            });
        }
    }
    let mut conflicts = Vec::new();
    for ((side, drainage_node_id), channels) in maxima {
        let mut area = channels[0].clone().unwrap_or_default();
        let mut line = channels[1].clone().unwrap_or_default();
        area.sort_unstable();
        area.dedup();
        line.sort_unstable();
        line.dedup();
        if area != line {
            conflicts.push(MetricConflictV0 {
                side,
                drainage_node_id,
                area_maximum_ids: area,
                line_maximum_ids: line,
            });
        }
    }
    Ok(conflicts)
}

#[allow(clippy::too_many_arguments)]
pub fn build_topology_records_v0(
    side: PacketSideV0,
    family: ObjectFamilyV0,
    channel: AssignmentChannelV0,
    source_edges: &[TopologyEdgeInputV0],
    opposite_topology: &[TopologyObjectInputV0],
    assignments: &[AssignmentV0],
    best_components: &[BestComponentV0],
    declared_portal_ids: &[u32],
) -> Result<Vec<TopologyV0>, CorrespondenceErrorV0> {
    match (family, channel) {
        (ObjectFamilyV0::Highland, AssignmentChannelV0::HighlandExclusiveArea)
        | (ObjectFamilyV0::DrainageNode, AssignmentChannelV0::DrainageExclusiveArea)
        | (ObjectFamilyV0::DrainageNode, AssignmentChannelV0::DrainageLine) => {}
        _ => return Err(CorrespondenceErrorV0::UnregisteredConfiguration),
    }
    let declared_portals = declared_portal_ids.iter().copied().collect::<BTreeSet<_>>();
    if declared_portals.len() != declared_portal_ids.len() {
        return Err(CorrespondenceErrorV0::UnregisteredConfiguration);
    }
    let mut topology = BTreeMap::new();
    for object in opposite_topology {
        validate_topology_target(family, object.target, &declared_portals)?;
        if topology.insert(object.object_id, object.target).is_some() {
            return Err(CorrespondenceErrorV0::DuplicateTopologyObject(
                object.object_id,
            ));
        }
    }
    let mut maximum_sets = BTreeMap::new();
    for assignment in assignments.iter().filter(|assignment| {
        assignment.side == side && assignment.family == family && assignment.channel == channel
    }) {
        let mut maximum_partner_ids = assignment.maximum_partner_ids.clone();
        maximum_partner_ids.sort_unstable();
        maximum_partner_ids.dedup();
        if maximum_sets
            .insert(assignment.object_id, maximum_partner_ids)
            .is_some()
        {
            return Err(CorrespondenceErrorV0::DuplicateAssignment {
                side,
                id: assignment.object_id,
                channel,
            });
        }
    }
    let mut seen_edges = BTreeSet::new();
    let mut edges = source_edges.to_vec();
    edges.sort_by(|a, b| {
        a.from_id
            .cmp(&b.from_id)
            .then_with(|| topology_target_cmp(a.target, b.target))
    });
    let mut records = Vec::with_capacity(edges.len());
    for edge in edges {
        validate_topology_target(family, edge.target, &declared_portals)?;
        if !seen_edges.insert(edge.from_id) {
            return Err(CorrespondenceErrorV0::DuplicateTopologyEdge(edge.from_id));
        }
        if edge.hierarchy_ambiguous {
            records.push(TopologyV0 {
                side,
                family,
                channel,
                from_id: edge.from_id,
                target: edge.target,
                availability: TopologyAvailabilityV0::HierarchyAmbiguous,
                mapped_adjacency: None,
                endpoints_in_same_best_component: None,
            });
            continue;
        }
        let from_maxima = maximum_sets
            .get(&edge.from_id)
            .filter(|ids| !ids.is_empty());
        let object_target = topology_object_id(family, edge.target);
        let target_maxima =
            object_target.map(|id| maximum_sets.get(&id).filter(|ids| !ids.is_empty()));
        let missing_endpoint =
            from_maxima.is_none() || object_target.is_some() && target_maxima.flatten().is_none();
        if missing_endpoint {
            records.push(TopologyV0 {
                side,
                family,
                channel,
                from_id: edge.from_id,
                target: edge.target,
                availability: TopologyAvailabilityV0::NoMappedEndpoint,
                mapped_adjacency: None,
                endpoints_in_same_best_component: None,
            });
            continue;
        }
        let from_maxima = from_maxima.expect("mapped endpoint checked");
        let mut matching = 0usize;
        let total;
        if let Some(target_id) = object_target {
            let target_maxima = target_maxima
                .flatten()
                .expect("mapped object endpoint checked");
            total = from_maxima.len().checked_mul(target_maxima.len()).ok_or(
                CorrespondenceErrorV0::NumericalFailure("topology product overflow"),
            )?;
            for &mapped_from in from_maxima {
                let mapped_target = topology
                    .get(&mapped_from)
                    .copied()
                    .ok_or(CorrespondenceErrorV0::MissingTopologyObject(mapped_from))?;
                for &mapped_to in target_maxima {
                    if mapped_target == topology_object_target(family, mapped_to) {
                        matching += 1;
                    }
                }
            }
            records.push(TopologyV0 {
                side,
                family,
                channel,
                from_id: edge.from_id,
                target: edge.target,
                availability: TopologyAvailabilityV0::Available,
                mapped_adjacency: Some(adjacency_category(matching, total)),
                endpoints_in_same_best_component: Some(endpoints_share_component(
                    side,
                    family,
                    channel,
                    edge.from_id,
                    target_id,
                    best_components,
                )),
            });
        } else {
            total = from_maxima.len();
            for &mapped_from in from_maxima {
                let mapped_target = topology
                    .get(&mapped_from)
                    .copied()
                    .ok_or(CorrespondenceErrorV0::MissingTopologyObject(mapped_from))?;
                if mapped_target == edge.target {
                    matching += 1;
                }
            }
            records.push(TopologyV0 {
                side,
                family,
                channel,
                from_id: edge.from_id,
                target: edge.target,
                availability: TopologyAvailabilityV0::Available,
                mapped_adjacency: Some(adjacency_category(matching, total)),
                endpoints_in_same_best_component: None,
            });
        }
    }
    Ok(records)
}

fn validate_topology_target(
    family: ObjectFamilyV0,
    target: TopologyTargetV0,
    declared_portals: &BTreeSet<u32>,
) -> Result<(), CorrespondenceErrorV0> {
    match (family, target) {
        (ObjectFamilyV0::Highland, TopologyTargetV0::Highland(_))
        | (ObjectFamilyV0::Highland, TopologyTargetV0::HighlandRoot)
        | (ObjectFamilyV0::DrainageNode, TopologyTargetV0::DrainageNode(_)) => Ok(()),
        (ObjectFamilyV0::DrainageNode, TopologyTargetV0::Portal(id)) => {
            if declared_portals.contains(&id) {
                Ok(())
            } else {
                Err(CorrespondenceErrorV0::UndeclaredPortal(id))
            }
        }
        _ => Err(CorrespondenceErrorV0::InvalidTopologyTarget(target)),
    }
}

fn topology_object_id(family: ObjectFamilyV0, target: TopologyTargetV0) -> Option<u32> {
    match (family, target) {
        (ObjectFamilyV0::Highland, TopologyTargetV0::Highland(id))
        | (ObjectFamilyV0::DrainageNode, TopologyTargetV0::DrainageNode(id)) => Some(id),
        _ => None,
    }
}

fn topology_object_target(family: ObjectFamilyV0, id: u32) -> TopologyTargetV0 {
    match family {
        ObjectFamilyV0::Highland => TopologyTargetV0::Highland(id),
        ObjectFamilyV0::DrainageNode => TopologyTargetV0::DrainageNode(id),
    }
}

fn topology_target_cmp(a: TopologyTargetV0, b: TopologyTargetV0) -> Ordering {
    fn key(value: TopologyTargetV0) -> (u8, u32) {
        match value {
            TopologyTargetV0::Highland(id) => (0, id),
            TopologyTargetV0::DrainageNode(id) => (1, id),
            TopologyTargetV0::Portal(id) => (2, id),
            TopologyTargetV0::HighlandRoot => (3, 0),
        }
    }
    key(a).cmp(&key(b))
}

fn adjacency_category(matching: usize, total: usize) -> MappedAdjacencyV0 {
    if matching == total {
        MappedAdjacencyV0::All
    } else if matching == 0 {
        MappedAdjacencyV0::None
    } else {
        MappedAdjacencyV0::Some
    }
}

fn endpoints_share_component(
    side: PacketSideV0,
    family: ObjectFamilyV0,
    channel: AssignmentChannelV0,
    from_id: u32,
    target_id: u32,
    components: &[BestComponentV0],
) -> bool {
    components.iter().any(|component| {
        component.channel == channel
            && component.members.contains(&BestMemberV0 {
                side,
                family,
                object_id: from_id,
            })
            && component.members.contains(&BestMemberV0 {
                side,
                family,
                object_id: target_id,
            })
    })
}

fn validate_assignment_objects(
    side: PacketSideV0,
    objects: &[AssignmentObjectInputV0],
) -> Result<BTreeMap<u32, &AssignmentObjectInputV0>, CorrespondenceErrorV0> {
    let mut result = BTreeMap::new();
    for object in objects {
        if !object.anchor_km.is_finite()
            || !object.object_measure.is_finite()
            || (object.support_status == SupportStatusV0::Eligible && object.object_measure <= 0.0)
            || object.object_measure < 0.0
        {
            return Err(CorrespondenceErrorV0::InvalidObjectMeasure {
                side,
                id: object.object_id,
            });
        }
        if result.insert(object.object_id, object).is_some() {
            return Err(CorrespondenceErrorV0::DuplicateObject {
                side,
                id: object.object_id,
            });
        }
    }
    Ok(result)
}

fn build_assignment(
    side: PacketSideV0,
    family: ObjectFamilyV0,
    channel: AssignmentChannelV0,
    object: &AssignmentObjectInputV0,
    mut scores: Vec<(u32, f64)>,
) -> Result<AssignmentV0, CorrespondenceErrorV0> {
    scores.sort_by_key(|value| value.0);
    if object.support_status != SupportStatusV0::Eligible {
        if !scores.is_empty() {
            return Err(CorrespondenceErrorV0::InvalidPositiveScore);
        }
        return Ok(AssignmentV0 {
            side,
            family,
            object_id: object.object_id,
            channel,
            support_status: object.support_status,
            positive_partner_ids: Vec::new(),
            maximum_partner_ids: Vec::new(),
            best_score: None,
            second_distinct_score: None,
            normalized_margin: None,
            exact_best_tie: false,
        });
    }
    if scores.is_empty() {
        return Ok(AssignmentV0 {
            side,
            family,
            object_id: object.object_id,
            channel,
            support_status: SupportStatusV0::NoPositiveOverlap,
            positive_partner_ids: Vec::new(),
            maximum_partner_ids: Vec::new(),
            best_score: None,
            second_distinct_score: None,
            normalized_margin: None,
            exact_best_tie: false,
        });
    }
    let best = scores
        .iter()
        .map(|value| value.1)
        .max_by(f64::total_cmp)
        .expect("positive scores are nonempty");
    let maximum_partner_ids = scores
        .iter()
        .filter(|value| value.1.total_cmp(&best) == Ordering::Equal)
        .map(|value| value.0)
        .collect::<Vec<_>>();
    let second_distinct = scores
        .iter()
        .map(|value| value.1)
        .filter(|score| score.total_cmp(&best) == Ordering::Less)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let exact_best_tie = maximum_partner_ids.len() > 1;
    let normalized_margin = if exact_best_tie {
        0.0
    } else {
        (best - second_distinct) / object.object_measure
    };
    if !normalized_margin.is_finite() || normalized_margin < 0.0 {
        return Err(CorrespondenceErrorV0::NumericalFailure("assignment margin"));
    }
    Ok(AssignmentV0 {
        side,
        family,
        object_id: object.object_id,
        channel,
        support_status: SupportStatusV0::Eligible,
        positive_partner_ids: scores.iter().map(|value| value.0).collect(),
        maximum_partner_ids,
        best_score: Some(best),
        second_distinct_score: Some(second_distinct),
        normalized_margin: Some(normalized_margin),
        exact_best_tie,
    })
}

fn build_best_components(
    family: ObjectFamilyV0,
    channel: AssignmentChannelV0,
    assignments: &[AssignmentV0],
    source: &BTreeMap<u32, &AssignmentObjectInputV0>,
    target: &BTreeMap<u32, &AssignmentObjectInputV0>,
) -> Result<Vec<BestComponentV0>, CorrespondenceErrorV0> {
    let mut adjacency: BTreeMap<BestMemberV0, BTreeSet<BestMemberV0>> = BTreeMap::new();
    for assignment in assignments {
        for &partner in &assignment.maximum_partner_ids {
            let member = BestMemberV0 {
                side: assignment.side,
                family,
                object_id: assignment.object_id,
            };
            let opposite = BestMemberV0 {
                side: match assignment.side {
                    PacketSideV0::Source => PacketSideV0::Target,
                    PacketSideV0::Target => PacketSideV0::Source,
                },
                family,
                object_id: partner,
            };
            adjacency.entry(member).or_default().insert(opposite);
            adjacency.entry(opposite).or_default().insert(member);
        }
    }
    let mut visited = BTreeSet::new();
    let mut keyed_components = Vec::new();
    for &start in adjacency.keys() {
        if visited.contains(&start) {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut members = Vec::new();
        visited.insert(start);
        while let Some(member) = queue.pop_front() {
            members.push(member);
            for &neighbor in &adjacency[&member] {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        members.sort();
        let source_count = members
            .iter()
            .filter(|member| member.side == PacketSideV0::Source)
            .count();
        let target_count = members.len() - source_count;
        let kind = match (source_count, target_count) {
            (1, 1) => ComponentKindV0::OneToOneBest,
            (1, _) => ComponentKindV0::OneToManyBest,
            (_, 1) => ComponentKindV0::ManyToOneBest,
            _ => ComponentKindV0::ManyToManyBest,
        };
        let key = members
            .iter()
            .map(|member| {
                let object = match member.side {
                    PacketSideV0::Source => source.get(&member.object_id),
                    PacketSideV0::Target => target.get(&member.object_id),
                }
                .copied()
                .ok_or(CorrespondenceErrorV0::UnknownObject {
                    side: member.side,
                    id: member.object_id,
                })?;
                Ok((object.anchor_km, *member))
            })
            .collect::<Result<Vec<_>, CorrespondenceErrorV0>>()?
            .into_iter()
            .min_by(|a, b| {
                point_cmp(a.0, b.0)
                    .then_with(|| a.1.side.cmp(&b.1.side))
                    .then_with(|| a.1.object_id.cmp(&b.1.object_id))
            })
            .expect("best component has members");
        keyed_components.push((
            key,
            BestComponentV0 {
                channel,
                kind,
                members,
            },
        ));
    }
    keyed_components.sort_by(|a, b| {
        point_cmp(a.0 .0, b.0 .0)
            .then_with(|| a.0 .1.side.cmp(&b.0 .1.side))
            .then_with(|| a.0 .1.object_id.cmp(&b.0 .1.object_id))
            .then_with(|| a.1.members.cmp(&b.1.members))
    });
    Ok(keyed_components.into_iter().map(|value| value.1).collect())
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    fn assignment(
        family: ObjectFamilyV0,
        channel: AssignmentChannelV0,
        id: u32,
        maxima: &[u32],
    ) -> AssignmentV0 {
        AssignmentV0 {
            side: PacketSideV0::Source,
            family,
            object_id: id,
            channel,
            support_status: SupportStatusV0::Eligible,
            positive_partner_ids: maxima.to_vec(),
            maximum_partner_ids: maxima.to_vec(),
            best_score: Some(1.0),
            second_distinct_score: Some(0.0),
            normalized_margin: Some(1.0),
            exact_best_tie: maxima.len() > 1,
        }
    }

    fn drainage_edge_record(
        maxima_a: &[u32],
        maxima_b: &[u32],
        opposite: &[TopologyObjectInputV0],
    ) -> TopologyV0 {
        build_topology_records_v0(
            PacketSideV0::Source,
            ObjectFamilyV0::DrainageNode,
            AssignmentChannelV0::DrainageLine,
            &[TopologyEdgeInputV0 {
                from_id: 0,
                target: TopologyTargetV0::DrainageNode(1),
                hierarchy_ambiguous: false,
            }],
            opposite,
            &[
                assignment(
                    ObjectFamilyV0::DrainageNode,
                    AssignmentChannelV0::DrainageLine,
                    0,
                    maxima_a,
                ),
                assignment(
                    ObjectFamilyV0::DrainageNode,
                    AssignmentChannelV0::DrainageLine,
                    1,
                    maxima_b,
                ),
            ],
            &[],
            &[],
        )
        .unwrap()
        .remove(0)
    }

    #[test]
    fn topology_cartesian_products_report_all_some_and_none() {
        let all = drainage_edge_record(
            &[10],
            &[11],
            &[TopologyObjectInputV0 {
                object_id: 10,
                target: TopologyTargetV0::DrainageNode(11),
            }],
        );
        assert_eq!(all.mapped_adjacency, Some(MappedAdjacencyV0::All));
        assert_eq!(all.endpoints_in_same_best_component, Some(false));

        let some = drainage_edge_record(
            &[10, 11],
            &[12],
            &[
                TopologyObjectInputV0 {
                    object_id: 10,
                    target: TopologyTargetV0::DrainageNode(12),
                },
                TopologyObjectInputV0 {
                    object_id: 11,
                    target: TopologyTargetV0::DrainageNode(13),
                },
            ],
        );
        assert_eq!(some.mapped_adjacency, Some(MappedAdjacencyV0::Some));

        let none = drainage_edge_record(
            &[10],
            &[11],
            &[TopologyObjectInputV0 {
                object_id: 10,
                target: TopologyTargetV0::DrainageNode(99),
            }],
        );
        assert_eq!(none.mapped_adjacency, Some(MappedAdjacencyV0::None));
    }

    #[test]
    fn topology_portal_root_and_undeclared_targets_are_semantic() {
        let portal = build_topology_records_v0(
            PacketSideV0::Source,
            ObjectFamilyV0::DrainageNode,
            AssignmentChannelV0::DrainageLine,
            &[TopologyEdgeInputV0 {
                from_id: 0,
                target: TopologyTargetV0::Portal(7),
                hierarchy_ambiguous: false,
            }],
            &[TopologyObjectInputV0 {
                object_id: 10,
                target: TopologyTargetV0::Portal(7),
            }],
            &[assignment(
                ObjectFamilyV0::DrainageNode,
                AssignmentChannelV0::DrainageLine,
                0,
                &[10],
            )],
            &[],
            &[7],
        )
        .unwrap();
        assert_eq!(portal[0].mapped_adjacency, Some(MappedAdjacencyV0::All));
        assert_eq!(portal[0].endpoints_in_same_best_component, None);

        let root = build_topology_records_v0(
            PacketSideV0::Source,
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea,
            &[TopologyEdgeInputV0 {
                from_id: 0,
                target: TopologyTargetV0::HighlandRoot,
                hierarchy_ambiguous: false,
            }],
            &[TopologyObjectInputV0 {
                object_id: 10,
                target: TopologyTargetV0::HighlandRoot,
            }],
            &[assignment(
                ObjectFamilyV0::Highland,
                AssignmentChannelV0::HighlandExclusiveArea,
                0,
                &[10],
            )],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(root[0].mapped_adjacency, Some(MappedAdjacencyV0::All));

        let unavailable = build_topology_records_v0(
            PacketSideV0::Source,
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea,
            &[
                TopologyEdgeInputV0 {
                    from_id: 0,
                    target: TopologyTargetV0::Highland(1),
                    hierarchy_ambiguous: false,
                },
                TopologyEdgeInputV0 {
                    from_id: 2,
                    target: TopologyTargetV0::Highland(3),
                    hierarchy_ambiguous: true,
                },
            ],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            unavailable[0].availability,
            TopologyAvailabilityV0::NoMappedEndpoint
        );
        assert_eq!(unavailable[0].mapped_adjacency, None);
        assert_eq!(
            unavailable[1].availability,
            TopologyAvailabilityV0::HierarchyAmbiguous
        );
        assert_eq!(unavailable[1].mapped_adjacency, None);

        assert_eq!(
            build_topology_records_v0(
                PacketSideV0::Source,
                ObjectFamilyV0::DrainageNode,
                AssignmentChannelV0::DrainageLine,
                &[TopologyEdgeInputV0 {
                    from_id: 0,
                    target: TopologyTargetV0::Portal(8),
                    hierarchy_ambiguous: false,
                }],
                &[],
                &[],
                &[],
                &[7],
            ),
            Err(CorrespondenceErrorV0::UndeclaredPortal(8))
        );
    }
}
