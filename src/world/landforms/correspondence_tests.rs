//! Manufactured public-kernel gates for O0b correspondence.
//!
//! The little analytic types in this file are deliberately independent of the
//! production implementation.  They make the preregistered answers readable
//! at the call site and prevent a production helper from becoming its own
//! oracle.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
}

impl Rect {
    const fn new(x0: f64, x1: f64, y0: f64, y1: f64) -> Self {
        Self { x0, x1, y0, y1 }
    }

    fn area(self) -> f64 {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }

    fn intersection_area(self, other: Self) -> f64 {
        let width = (self.x1.min(other.x1) - self.x0.max(other.x0)).max(0.0);
        let height = (self.y1.min(other.y1) - self.y0.max(other.y0)).max(0.0);
        width * height
    }

    fn polygon(self) -> Vec<DVec3> {
        vec![
            DVec3::new(self.x0, self.y0, 0.0),
            DVec3::new(self.x1, self.y0, 0.0),
            DVec3::new(self.x1, self.y1, 0.0),
            DVec3::new(self.x0, self.y1, 0.0),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AreaOracle {
    intersection: f64,
    union: f64,
    source_coverage: f64,
    target_coverage: f64,
    jaccard: f64,
    dice: f64,
}

fn rectangle_oracle(source: Rect, target: Rect) -> Option<AreaOracle> {
    let intersection = source.intersection_area(target);
    if intersection == 0.0 {
        return None;
    }
    let source_area = source.area();
    let target_area = target.area();
    let union = source_area + target_area - intersection;
    Some(AreaOracle {
        intersection,
        union,
        source_coverage: intersection / source_area,
        target_coverage: intersection / target_area,
        jaccard: intersection / union,
        dice: 2.0 * intersection / (source_area + target_area),
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CollinearCoverageOracle {
    source_covered: f64,
    target_covered: f64,
}

/// Independent oracle for the collinear finite-capsule fixtures.  The target
/// interval expanded by the pair radius is intersected with the source, and
/// vice versa.  This is intentionally not the general segment algorithm.
fn collinear_capsule_oracle(
    source: [f64; 2],
    target: [f64; 2],
    pair_radius: f64,
) -> Option<CollinearCoverageOracle> {
    let source_covered =
        (source[1].min(target[1] + pair_radius) - source[0].max(target[0] - pair_radius)).max(0.0);
    let target_covered =
        (target[1].min(source[1] + pair_radius) - target[0].max(source[0] - pair_radius)).max(0.0);
    (source_covered > 0.0 && target_covered > 0.0).then_some(CollinearCoverageOracle {
        source_covered,
        target_covered,
    })
}

fn registered_rectangle_cells() -> (Vec<Rect>, Vec<Rect>, Vec<Rect>) {
    let source_a = vec![
        Rect::new(0.0, 50.0, 0.0, 50.0),
        Rect::new(50.0, 100.0, 0.0, 50.0),
    ];
    let target_a = vec![
        Rect::new(0.0, 100.0, 0.0, 25.0),
        Rect::new(0.0, 100.0, 25.0, 50.0),
    ];
    let target_b = vec![
        Rect::new(25.0, 75.0, 0.0, 50.0),
        Rect::new(75.0, 125.0, 0.0, 50.0),
    ];
    (source_a, target_a, target_b)
}

#[test]
fn o0b_preregistered_rectangle_oracles_are_exact() {
    let a = Rect::new(0.0, 100.0, 0.0, 50.0);
    let b = Rect::new(25.0, 125.0, 0.0, 50.0);
    assert_eq!(
        rectangle_oracle(a, a),
        Some(AreaOracle {
            intersection: 5_000.0,
            union: 5_000.0,
            source_coverage: 1.0,
            target_coverage: 1.0,
            jaccard: 1.0,
            dice: 1.0,
        })
    );
    assert_eq!(
        rectangle_oracle(a, b),
        Some(AreaOracle {
            intersection: 3_750.0,
            union: 6_250.0,
            source_coverage: 0.75,
            target_coverage: 0.75,
            jaccard: 0.60,
            dice: 0.75,
        })
    );
    assert_eq!(
        rectangle_oracle(a, Rect::new(100.0, 200.0, 0.0, 50.0)),
        None
    );
    assert_eq!(
        rectangle_oracle(a, Rect::new(101.0, 201.0, 0.0, 50.0)),
        None
    );
    let sliver = rectangle_oracle(a, Rect::new(99.999, 199.999, 0.0, 50.0)).unwrap();
    assert!((sliver.intersection - 0.05).abs() <= 1.0e-12);
    assert!((sliver.source_coverage - 0.00001).abs() <= 1.0e-15);
}

#[test]
fn o0b_preregistered_cell_tilings_have_frozen_contributions() {
    let (source, target_same, target_b) = registered_rectangle_cells();
    let same = source
        .iter()
        .flat_map(|&a| target_same.iter().map(move |&b| a.intersection_area(b)))
        .filter(|area| *area > 0.0)
        .collect::<Vec<_>>();
    assert_eq!(same, vec![1_250.0; 4]);
    assert_eq!(same.iter().sum::<f64>(), 5_000.0);

    let shifted = source
        .iter()
        .flat_map(|&a| target_b.iter().map(move |&b| a.intersection_area(b)))
        .filter(|area| *area > 0.0)
        .collect::<Vec<_>>();
    assert_eq!(shifted, vec![1_250.0; 3]);
    assert_eq!(shifted.iter().sum::<f64>(), 3_750.0);

    // Keep the polygon representation exercised so production fixtures use
    // the exact CCW vertex convention rather than direct object rectangles.
    assert!(source
        .iter()
        .chain(&target_same)
        .chain(&target_b)
        .all(|rect| rect.polygon().len() == 4));
}

#[test]
fn o0b_preregistered_best_graph_matrix_is_exact() {
    let source = [
        Rect::new(0.0, 60.0, 0.0, 100.0),
        Rect::new(60.0, 100.0, 0.0, 100.0),
    ];
    let target = [
        Rect::new(0.0, 100.0, 0.0, 60.0),
        Rect::new(0.0, 100.0, 60.0, 100.0),
    ];
    let table = source.map(|a| target.map(|b| a.intersection_area(b)));
    assert_eq!(table, [[3_600.0, 2_400.0], [2_400.0, 1_600.0]]);
    assert_eq!(table[0][0].max(table[0][1]), 3_600.0);
    assert_eq!(table[1][0].max(table[1][1]), 2_400.0);
    assert_eq!(table[0][0].max(table[1][0]), 3_600.0);
    assert_eq!(table[0][1].max(table[1][1]), 2_400.0);
}

#[test]
fn o0b_preregistered_collinear_capsule_oracles_include_endpoint_caps() {
    assert_eq!(
        collinear_capsule_oracle([0.0, 100.0], [0.0, 100.0], 1.0),
        Some(CollinearCoverageOracle {
            source_covered: 100.0,
            target_covered: 100.0,
        })
    );
    assert_eq!(
        collinear_capsule_oracle([0.0, 100.0], [25.0, 75.0], 1.0),
        Some(CollinearCoverageOracle {
            source_covered: 52.0,
            target_covered: 50.0,
        })
    );
    assert_eq!(
        collinear_capsule_oracle([0.0, 100.0], [0.0, 60.0], 1.0),
        Some(CollinearCoverageOracle {
            source_covered: 61.0,
            target_covered: 60.0,
        })
    );
    assert_eq!(
        collinear_capsule_oracle([0.0, 100.0], [60.0, 100.0], 1.0),
        Some(CollinearCoverageOracle {
            source_covered: 41.0,
            target_covered: 40.0,
        })
    );
}

fn polygons(rectangles: &[Rect]) -> Vec<Vec<DVec3>> {
    rectangles.iter().map(|rect| rect.polygon()).collect()
}

fn rigid_transform(point: DVec3) -> DVec3 {
    let translated = point + DVec3::new(17.0, -23.0, 0.0);
    let (sin, cos) = (std::f64::consts::PI / 6.0).sin_cos();
    DVec3::new(
        cos * translated.x - sin * translated.y,
        sin * translated.x + cos * translated.y,
        0.0,
    )
}

fn inverse_rigid_transform(point: DVec3) -> DVec3 {
    let (sin, cos) = (std::f64::consts::PI / 6.0).sin_cos();
    DVec3::new(
        cos * point.x + sin * point.y - 17.0,
        -sin * point.x + cos * point.y + 23.0,
        0.0,
    )
}

fn transform_polygons(input: &[Vec<DVec3>]) -> Vec<Vec<DVec3>> {
    input
        .iter()
        .map(|polygon| polygon.iter().copied().map(rigid_transform).collect())
        .collect()
}

fn transform_segments(input: &[LineSegmentInputV0]) -> Vec<LineSegmentInputV0> {
    input
        .iter()
        .map(|segment| LineSegmentInputV0 {
            endpoints_km: segment.endpoints_km.map(rigid_transform),
            ..*segment
        })
        .collect()
}

fn line_segment(x0: f64, x1: f64, y: f64, local_radius_km: f64) -> LineSegmentInputV0 {
    LineSegmentInputV0 {
        endpoints_km: [DVec3::new(x0, y, 0.0), DVec3::new(x1, y, 0.0)],
        measure_length_km: x1 - x0,
        local_radius_km,
    }
}

#[test]
fn o0b_production_area_kernel_matches_exact_rectangle_and_cell_tiling_oracles() {
    let (source_cells, target_same_cells, target_b_cells) = registered_rectangle_cells();
    let same = build_area_pair_v0(
        7,
        9,
        AreaSupportV0::Exclusive,
        &polygons(&source_cells),
        &polygons(&target_same_cells),
        1.0e-8,
        1.0e-10,
    )
    .unwrap()
    .unwrap();
    assert_eq!(same.intersection_area_km2, 5_000.0);
    assert_eq!(same.source_area_km2, 5_000.0);
    assert_eq!(same.target_area_km2, 5_000.0);
    assert_eq!(same.union_area_km2, 5_000.0);
    assert_eq!(same.source_coverage, 1.0);
    assert_eq!(same.target_coverage, 1.0);
    assert_eq!(same.jaccard, 1.0);
    assert_eq!(same.dice, 1.0);

    let shifted = build_area_pair_v0(
        7,
        11,
        AreaSupportV0::Exclusive,
        &polygons(&source_cells),
        &polygons(&target_b_cells),
        1.0e-8,
        1.0e-10,
    )
    .unwrap()
    .unwrap();
    assert_eq!(shifted.intersection_area_km2, 3_750.0);
    assert_eq!(shifted.union_area_km2, 6_250.0);
    assert_eq!(shifted.source_coverage, 0.75);
    assert_eq!(shifted.target_coverage, 0.75);
    assert_eq!(shifted.jaccard, 0.60);
    assert_eq!(shifted.dice, 0.75);
    let reversed = build_area_pair_v0(
        11,
        7,
        AreaSupportV0::Exclusive,
        &polygons(&target_b_cells),
        &polygons(&source_cells),
        1.0e-8,
        1.0e-10,
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        reversed.intersection_area_km2.to_bits(),
        shifted.intersection_area_km2.to_bits()
    );
    assert_eq!(
        reversed.union_area_km2.to_bits(),
        shifted.union_area_km2.to_bits()
    );
    assert_eq!(
        reversed.source_coverage.to_bits(),
        shifted.target_coverage.to_bits()
    );
    assert_eq!(
        reversed.target_coverage.to_bits(),
        shifted.source_coverage.to_bits()
    );
    assert_eq!(reversed.jaccard.to_bits(), shifted.jaccard.to_bits());
    assert_eq!(reversed.dice.to_bits(), shifted.dice.to_bits());
    let mut source_reenumerated = source_cells.clone();
    let mut target_reenumerated = target_b_cells.clone();
    source_reenumerated.reverse();
    target_reenumerated.reverse();
    let reenumerated = build_area_pair_v0(
        7,
        11,
        AreaSupportV0::Exclusive,
        &polygons(&source_reenumerated),
        &polygons(&target_reenumerated),
        1.0e-8,
        1.0e-10,
    )
    .unwrap()
    .unwrap();
    assert_eq!(reenumerated, shifted);

    let edge_contact = build_area_pair_v0(
        7,
        13,
        AreaSupportV0::Exclusive,
        &polygons(&source_cells),
        &polygons(&[Rect::new(100.0, 200.0, 0.0, 50.0)]),
        1.0e-8,
        1.0e-10,
    )
    .unwrap();
    assert!(edge_contact.is_none());

    let sliver = build_area_pair_v0(
        7,
        15,
        AreaSupportV0::Exclusive,
        &polygons(&source_cells),
        &polygons(&[Rect::new(99.999, 199.999, 0.0, 50.0)]),
        1.0e-8,
        1.0e-10,
    )
    .unwrap()
    .unwrap();
    assert!((sliver.intersection_area_km2 - 0.05).abs() <= 1.0e-12);
    assert!((sliver.source_coverage - 0.00001).abs() <= 1.0e-15);
}

#[test]
fn o0b_production_line_kernel_matches_finite_capsule_oracles() {
    let source = [line_segment(0.0, 100.0, 0.0, 0.5)];
    let partial = [line_segment(25.0, 75.0, 0.0, 0.5)];
    let row = build_line_pair_v0(1, 2, &source, &partial)
        .unwrap()
        .unwrap();
    assert_eq!(row.source_covered_length_km, 52.0);
    assert_eq!(row.target_covered_length_km, 50.0);
    assert_eq!(row.source_coverage, 0.52);
    assert_eq!(row.target_coverage, 1.0);
    let population = build_line_population_v0(
        &[LineObjectInputV0 {
            object_id: 1,
            segments: &source,
        }],
        &[LineObjectInputV0 {
            object_id: 2,
            segments: &partial,
        }],
    )
    .unwrap();
    assert_eq!(population.pairs, vec![row.clone()]);
    assert_eq!(population.segment_box_candidates, 1);
    assert_eq!(population.segment_pair_tests, 1);

    let split = [
        line_segment(0.0, 40.0, 0.0, 0.5),
        line_segment(40.0, 100.0, 0.0, 0.5),
    ];
    let row = build_line_pair_v0(1, 3, &source, &split).unwrap().unwrap();
    assert_eq!(row.source_coverage, 1.0);
    assert_eq!(row.target_coverage, 1.0);

    let left = build_line_pair_v0(1, 31, &source, &[line_segment(0.0, 60.0, 0.0, 0.5)])
        .unwrap()
        .unwrap();
    assert_eq!(left.source_covered_length_km, 61.0);
    assert_eq!(left.target_covered_length_km, 60.0);
    let right = build_line_pair_v0(1, 32, &source, &[line_segment(60.0, 100.0, 0.0, 0.5)])
        .unwrap()
        .unwrap();
    assert_eq!(right.source_covered_length_km, 41.0);
    assert_eq!(right.target_covered_length_km, 40.0);
    let reversed = build_line_pair_v0(31, 1, &[line_segment(0.0, 60.0, 0.0, 0.5)], &source)
        .unwrap()
        .unwrap();
    assert_eq!(
        reversed.source_covered_length_km.to_bits(),
        left.target_covered_length_km.to_bits()
    );
    assert_eq!(
        reversed.target_covered_length_km.to_bits(),
        left.source_covered_length_km.to_bits()
    );

    assert!(
        build_line_pair_v0(1, 4, &source, &[line_segment(0.0, 100.0, 1.001, 0.5)])
            .unwrap()
            .is_none()
    );
    let buffered = build_line_pair_v0(1, 5, &source, &[line_segment(0.0, 100.0, 0.75, 0.5)])
        .unwrap()
        .unwrap();
    assert_eq!(buffered.source_coverage, 1.0);
    assert_eq!(buffered.target_coverage, 1.0);

    let local_control_source = [line_segment(0.0, 100.0, 1.5, 0.5)];
    assert!(build_line_pair_v0(
        6,
        7,
        &local_control_source,
        &[line_segment(0.0, 100.0, 0.0, 0.5)],
    )
    .unwrap()
    .is_none());
    assert!(build_line_pair_v0(
        6,
        8,
        &local_control_source,
        &[line_segment(1000.0, 1010.0, 0.0, 100.0)],
    )
    .unwrap()
    .is_none());
}

#[test]
fn o0b_production_tie_free_geometry_is_rigid_transform_covariant() {
    let source = polygons(&registered_rectangle_cells().0);
    let target = polygons(&registered_rectangle_cells().2);
    let reference = build_area_pair_v0(
        1,
        2,
        AreaSupportV0::Exclusive,
        &source,
        &target,
        1.0e-8,
        1.0e-10,
    )
    .unwrap()
    .unwrap();
    let transformed = build_area_pair_v0(
        1,
        2,
        AreaSupportV0::Exclusive,
        &transform_polygons(&source),
        &transform_polygons(&target),
        1.0e-8,
        1.0e-10,
    )
    .unwrap()
    .unwrap();
    for (actual, expected) in [
        (
            transformed.intersection_area_km2,
            reference.intersection_area_km2,
        ),
        (transformed.source_area_km2, reference.source_area_km2),
        (transformed.target_area_km2, reference.target_area_km2),
        (transformed.union_area_km2, reference.union_area_km2),
    ] {
        assert!((actual - expected).abs() <= 1.0e-10 * expected.max(1.0));
    }
    assert!(
        inverse_rigid_transform(transformed.source_centroid_km)
            .distance(reference.source_centroid_km)
            <= 1.0e-8
    );
    assert!(
        inverse_rigid_transform(transformed.target_centroid_km)
            .distance(reference.target_centroid_km)
            <= 1.0e-8
    );

    let source_line = [line_segment(0.0, 100.0, 0.0, 0.5)];
    let target_line = [line_segment(25.0, 75.0, 0.0, 0.5)];
    let reference_line = build_line_pair_v0(1, 2, &source_line, &target_line)
        .unwrap()
        .unwrap();
    let transformed_line = build_line_pair_v0(
        1,
        2,
        &transform_segments(&source_line),
        &transform_segments(&target_line),
    )
    .unwrap()
    .unwrap();
    assert!(
        (transformed_line.source_covered_length_km - reference_line.source_covered_length_km).abs()
            <= 1.0e-8
    );
    assert!(
        (transformed_line.target_covered_length_km - reference_line.target_covered_length_km).abs()
            <= 1.0e-8
    );
    assert!(
        inverse_rigid_transform(transformed_line.source_anchor_km)
            .distance(reference_line.source_anchor_km)
            <= 1.0e-8
    );
    assert!(
        inverse_rigid_transform(transformed_line.target_anchor_km)
            .distance(reference_line.target_anchor_km)
            <= 1.0e-8
    );
}

#[test]
fn o0b_production_geometry_kernels_reject_malformed_inputs() {
    assert_eq!(
        convex_polygon_intersection_v0(
            &[
                DVec3::ZERO,
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(f64::NAN, 1.0, 0.0),
            ],
            &Rect::new(0.0, 1.0, 0.0, 1.0).polygon(),
            1.0e-8,
        ),
        Err(CorrespondenceErrorV0::NonFiniteGeometry)
    );
    assert_eq!(
        convex_polygon_intersection_v0(
            &[DVec3::ZERO, DVec3::X, DVec3::new(2.0, 0.0, 0.0)],
            &Rect::new(0.0, 1.0, 0.0, 1.0).polygon(),
            1.0e-8,
        ),
        Err(CorrespondenceErrorV0::InvalidPolygon)
    );

    let rectangle = Rect::new(0.0, 100.0, 0.0, 50.0).polygon();
    assert_eq!(
        build_area_pair_v0(
            1,
            2,
            AreaSupportV0::Exclusive,
            &[rectangle.clone(), rectangle.clone()],
            &[rectangle],
            1.0e-8,
            1.0e-10,
        ),
        Err(CorrespondenceErrorV0::AreaBoundFailure)
    );

    let degenerate = LineSegmentInputV0 {
        endpoints_km: [DVec3::ZERO, DVec3::ZERO],
        measure_length_km: 1.0,
        local_radius_km: 0.5,
    };
    assert_eq!(
        build_line_pair_v0(1, 2, &[degenerate], &[line_segment(0.0, 1.0, 0.0, 0.5)]),
        Err(CorrespondenceErrorV0::DegenerateSegment)
    );
    let invalid_radius = LineSegmentInputV0 {
        endpoints_km: [DVec3::ZERO, DVec3::X],
        measure_length_km: 1.0,
        local_radius_km: f64::INFINITY,
    };
    assert_eq!(
        build_line_pair_v0(1, 2, &[invalid_radius], &[line_segment(0.0, 1.0, 0.0, 0.5)]),
        Err(CorrespondenceErrorV0::InvalidRadiusOrMeasure)
    );
    let nonplanar = LineSegmentInputV0 {
        endpoints_km: [DVec3::ZERO, DVec3::new(1.0, 0.0, 1.0)],
        measure_length_km: 1.0,
        local_radius_km: 0.5,
    };
    assert_eq!(
        build_line_pair_v0(1, 2, &[nonplanar], &[line_segment(0.0, 1.0, 0.0, 0.5)]),
        Err(CorrespondenceErrorV0::NonFiniteGeometry)
    );
}

fn assignment_object(id: u32, measure: f64, x: f64) -> AssignmentObjectInputV0 {
    AssignmentObjectInputV0 {
        object_id: id,
        object_measure: measure,
        anchor_km: DVec3::new(x, 0.0, 0.0),
        support_status: SupportStatusV0::Eligible,
    }
}

fn score(source_id: u32, target_id: u32, value: f64) -> PositiveScoreV0 {
    PositiveScoreV0 {
        source_id,
        target_id,
        source_score: value,
        target_score: value,
    }
}

fn assignment<'a>(
    output: &'a AssignmentKernelOutputV0,
    side: PacketSideV0,
    id: u32,
) -> &'a AssignmentV0 {
    output
        .assignments
        .iter()
        .find(|row| row.side == side && row.object_id == id)
        .unwrap()
}

#[test]
fn o0b_production_assignment_kernel_preserves_cardinality_ties_and_nulls() {
    let source = [assignment_object(1, 5_000.0, 0.0)];
    let targets = [
        assignment_object(10, 3_000.0, 10.0),
        assignment_object(11, 2_000.0, 20.0),
    ];
    let output = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &source,
        &targets,
        &[score(1, 10, 3_000.0), score(1, 11, 2_000.0)],
    )
    .unwrap();
    assert_eq!(output.best_components.len(), 1);
    assert_eq!(
        output.best_components[0].kind,
        ComponentKindV0::OneToManyBest
    );
    let source_assignment = assignment(&output, PacketSideV0::Source, 1);
    assert_eq!(source_assignment.maximum_partner_ids, vec![10]);
    assert!(!source_assignment.exact_best_tie);
    assert_eq!(source_assignment.normalized_margin, Some(0.2));

    let reversed_scores = [score(10, 1, 3_000.0), score(11, 1, 2_000.0)];
    let reversed = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &targets,
        &source,
        &reversed_scores,
    )
    .unwrap();
    assert_eq!(reversed.best_components.len(), 1);
    assert_eq!(
        reversed.best_components[0].kind,
        ComponentKindV0::ManyToOneBest
    );

    let tied_targets = [
        assignment_object(20, 2_500.0, 10.0),
        assignment_object(21, 2_500.0, 20.0),
    ];
    let tied = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &source,
        &tied_targets,
        &[score(1, 20, 2_500.0), score(1, 21, 2_500.0)],
    )
    .unwrap();
    let tied_source = assignment(&tied, PacketSideV0::Source, 1);
    assert_eq!(tied_source.maximum_partner_ids, vec![20, 21]);
    assert!(tied_source.exact_best_tie);
    assert_eq!(tied_source.normalized_margin, Some(0.0));

    let null = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &source,
        &[assignment_object(30, 100.0, 30.0)],
        &[],
    )
    .unwrap();
    assert!(null.best_components.is_empty());
    for row in &null.assignments {
        assert_eq!(row.support_status, SupportStatusV0::NoPositiveOverlap);
        assert!(row.best_score.is_none());
        assert!(row.normalized_margin.is_none());
        assert!(row.maximum_partner_ids.is_empty());
    }
}

#[test]
fn o0b_production_assignment_kernel_builds_exact_many_to_many_component() {
    let sources = [
        assignment_object(1, 6_000.0, 0.0),
        assignment_object(2, 4_000.0, 1.0),
    ];
    let targets = [
        assignment_object(10, 6_000.0, 10.0),
        assignment_object(11, 4_000.0, 11.0),
    ];
    let output = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &sources,
        &targets,
        &[
            score(1, 10, 3_600.0),
            score(1, 11, 2_400.0),
            score(2, 10, 2_400.0),
            score(2, 11, 1_600.0),
        ],
    )
    .unwrap();
    assert_eq!(output.best_components.len(), 1);
    assert_eq!(
        output.best_components[0].kind,
        ComponentKindV0::ManyToManyBest
    );
    assert_eq!(output.best_components[0].members.len(), 4);
    assert!(output.assignments.iter().all(|row| !row.exact_best_tie));

    let mut sources_reenumerated = sources;
    let mut targets_reenumerated = targets;
    let mut scores_reenumerated = [
        score(1, 10, 3_600.0),
        score(1, 11, 2_400.0),
        score(2, 10, 2_400.0),
        score(2, 11, 1_600.0),
    ];
    sources_reenumerated.reverse();
    targets_reenumerated.reverse();
    scores_reenumerated.reverse();
    let reenumerated = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &sources_reenumerated,
        &targets_reenumerated,
        &scores_reenumerated,
    )
    .unwrap();
    assert_eq!(reenumerated, output);
}

#[test]
fn o0b_production_metric_conflict_is_side_specific_and_keeps_both_channels() {
    let source = [assignment_object(1, 2_000.0, 0.0)];
    let targets = [
        assignment_object(10, 800.0, 10.0),
        assignment_object(11, 1_200.0, 20.0),
    ];
    let area = build_assignment_kernel_v0(
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageExclusiveArea,
        &source,
        &targets,
        &[score(1, 10, 800.0), score(1, 11, 1_200.0)],
    )
    .unwrap();
    let line = build_assignment_kernel_v0(
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
        &source,
        &targets,
        &[score(1, 10, 100.0)],
    )
    .unwrap();
    let assignments = area
        .assignments
        .iter()
        .chain(&line.assignments)
        .cloned()
        .collect::<Vec<_>>();
    let conflicts = build_metric_conflicts_v0(&assignments).unwrap();
    assert_eq!(conflicts.len(), 2);
    assert_eq!(conflicts[0].side, PacketSideV0::Source);
    assert_eq!(conflicts[0].drainage_node_id, 1);
    assert_eq!(conflicts[0].area_maximum_ids, vec![11]);
    assert_eq!(conflicts[0].line_maximum_ids, vec![10]);
    assert_eq!(conflicts[1].side, PacketSideV0::Target);
    assert_eq!(conflicts[1].drainage_node_id, 11);
    assert_eq!(conflicts[1].area_maximum_ids, vec![1]);
    assert!(conflicts[1].line_maximum_ids.is_empty());
}

#[test]
fn o0b_production_assignment_kernel_rejects_malformed_scores() {
    let source = [assignment_object(1, 1.0, 0.0)];
    let target = [assignment_object(2, 1.0, 1.0)];
    let duplicate = [score(1, 2, 1.0), score(1, 2, 1.0)];
    assert_eq!(
        build_assignment_kernel_v0(
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea,
            &source,
            &target,
            &duplicate,
        ),
        Err(CorrespondenceErrorV0::DuplicatePositivePair {
            source_id: 1,
            target_id: 2,
        })
    );
    assert_eq!(
        build_assignment_kernel_v0(
            ObjectFamilyV0::Highland,
            AssignmentChannelV0::HighlandExclusiveArea,
            &source,
            &target,
            &[score(1, 99, 1.0)],
        ),
        Err(CorrespondenceErrorV0::UnknownObject {
            side: PacketSideV0::Target,
            id: 99,
        })
    );
}

#[test]
fn o0b_production_ineligible_supports_have_no_numerical_assignment_or_component() {
    let source = [
        AssignmentObjectInputV0 {
            object_id: 1,
            object_measure: 0.0,
            anchor_km: DVec3::ZERO,
            support_status: SupportStatusV0::NoExclusiveSupport,
        },
        AssignmentObjectInputV0 {
            object_id: 2,
            object_measure: 100.0,
            anchor_km: DVec3::X,
            support_status: SupportStatusV0::HierarchyAmbiguousSupport,
        },
    ];
    let output = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &source,
        &[assignment_object(10, 100.0, 10.0)],
        &[],
    )
    .unwrap();
    assert!(output.best_components.is_empty());
    for row in output
        .assignments
        .iter()
        .filter(|row| row.side == PacketSideV0::Source)
    {
        assert!(matches!(
            row.support_status,
            SupportStatusV0::NoExclusiveSupport | SupportStatusV0::HierarchyAmbiguousSupport
        ));
        assert!(row.positive_partner_ids.is_empty());
        assert!(row.maximum_partner_ids.is_empty());
        assert!(row.best_score.is_none());
        assert!(row.second_distinct_score.is_none());
        assert!(row.normalized_margin.is_none());
        assert!(!row.exact_best_tie);
    }
}

#[test]
fn o0b_packet_population_extraction_partitions_reference_support() {
    let packet = super::packet_tests::assembled_asymmetric_y_packet_at(4.0);
    let highlands = extract_highland_population_v0(&packet).unwrap();
    assert_eq!(highlands.family, ObjectFamilyV0::Highland);
    assert!(!highlands.objects.is_empty());
    assert_eq!(highlands.cell_class.len(), packet.graph.cell_count());
    for object in &highlands.objects {
        assert!(!object.nested_cells.is_empty());
        assert!(object.nested_cells.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(object
            .exclusive_cells
            .windows(2)
            .all(|pair| pair[0] < pair[1]));
        assert!(object
            .exclusive_cells
            .iter()
            .all(|cell| object.nested_cells.binary_search(cell).is_ok()));
    }
    let classified_highland_cells = highlands
        .cell_class
        .iter()
        .filter(|class| !matches!(class, PacketCellClassV0::HighlandBackground))
        .count();
    let exclusive_highland_cells = highlands
        .objects
        .iter()
        .map(|object| object.exclusive_cells.len())
        .sum::<usize>();
    assert_eq!(classified_highland_cells, exclusive_highland_cells);

    let drainage = extract_drainage_population_v0(&packet).unwrap();
    assert_eq!(drainage.family, ObjectFamilyV0::DrainageNode);
    assert!(!drainage.objects.is_empty());
    assert_eq!(drainage.cell_class.len(), packet.graph.cell_count());
    assert!(drainage
        .objects
        .iter()
        .all(|object| object.status == SupportStatusV0::Eligible));
    assert!(drainage.cell_class.iter().all(|class| matches!(
        class,
        PacketCellClassV0::EligibleObject(_) | PacketCellClassV0::Portal(23)
    )));

    let reach_lines = extract_reference_reach_lines_v0(&packet).unwrap();
    assert_eq!(reach_lines.len(), drainage.objects.len());
    let reference_scale = packet
        .drainage
        .scales
        .iter()
        .find(|scale| scale.support_threshold_km2 == 2_000.0)
        .unwrap();
    for line in &reach_lines {
        let reach = reference_scale
            .reach_graph
            .reaches
            .iter()
            .find(|reach| reach.id == line.id)
            .unwrap();
        assert!(!line.segments.is_empty());
        assert!(line.segments.iter().all(|segment| {
            segment.endpoints_km[0] != segment.endpoints_km[1]
                && segment.measure_length_km > 0.0
                && segment.local_radius_km > 0.0
        }));
        let mut reproduced = 0.0;
        let mut correction = 0.0;
        for segment in &line.segments {
            let adjusted = segment.measure_length_km - correction;
            let next = reproduced + adjusted;
            correction = (next - reproduced) - adjusted;
            reproduced = next;
        }
        assert_eq!(reproduced.to_bits(), reach.physical_length_km.to_bits());
    }

    let first = build_object_correspondence_v0(&packet, &packet).unwrap();
    let repeated = build_object_correspondence_v0(&packet, &packet).unwrap();
    assert_eq!(repeated, first);
    assert_eq!(first.source_packet_hash, packet.derived_common_packet_hash);
    assert_eq!(first.target_packet_hash, packet.derived_common_packet_hash);
    assert_ne!(first.derived_correspondence_hash, 0);
    assert!(first.metric_conflicts.is_empty());
    assert!(first
        .assignment_records
        .iter()
        .filter(|row| row.support_status == SupportStatusV0::Eligible)
        .all(|row| row.maximum_partner_ids == vec![row.object_id]));
    assert!(first
        .best_components
        .iter()
        .all(|component| component.kind == ComponentKindV0::OneToOneBest));
    assert!(!first.topology_records.is_empty());
    assert!(first.topology_records.iter().all(|record| {
        record.availability == TopologyAvailabilityV0::Available
            && record.mapped_adjacency == Some(MappedAdjacencyV0::All)
            && match record.target {
                TopologyTargetV0::Highland(_) | TopologyTargetV0::DrainageNode(_) => {
                    record.endpoints_in_same_best_component.is_some()
                }
                TopologyTargetV0::Portal(_) | TopologyTargetV0::HighlandRoot => {
                    record.endpoints_in_same_best_component.is_none()
                }
            }
    }));
    assert!(first.context_records.iter().all(|context| {
        context.outside_domain_area_km2 == 0.0
            && context.background_area_km2 >= 0.0
            && context
                .ineligible_highland_areas
                .iter()
                .all(|entry| entry.area_km2 > 0.0)
            && context
                .portal_areas_km2
                .iter()
                .all(|entry| entry.area_km2 > 0.0)
    }));
    let full_cell_product = first
        .work_counts
        .source_cells
        .checked_mul(first.work_counts.target_cells)
        .unwrap();
    assert!(first.work_counts.cell_box_candidates < full_cell_product);
    assert_eq!(
        first.work_counts.polygon_clips,
        first.work_counts.cell_box_candidates
    );
    let full_segment_product = first
        .work_counts
        .source_segments
        .checked_mul(first.work_counts.target_segments)
        .unwrap();
    assert!(first.work_counts.segment_box_candidates < full_segment_product);
    assert_eq!(
        first.work_counts.segment_pair_tests,
        first.work_counts.segment_box_candidates
    );
    let bytes = object_correspondence_bytes_v0(&first).unwrap();
    let decoded = decode_object_correspondence_v0(&bytes).unwrap();
    assert_eq!(decoded, first);
    assert_eq!(object_correspondence_bytes_v0(&decoded).unwrap(), bytes);

    let mut tampered = first.clone();
    tampered.derived_correspondence_hash ^= 1;
    assert_eq!(
        object_correspondence_bytes_v0(&tampered),
        Err(PacketCorrespondenceErrorV0::Incompatible(
            "correspondence hash"
        ))
    );
    let bad_config = CorrespondenceConfigV0 {
        schema_version: "unregistered-o0b-test-config",
        ..CorrespondenceConfigV0::default()
    };
    assert_eq!(
        build_object_correspondence_with_config_v0(&packet, &packet, bad_config),
        Err(PacketCorrespondenceErrorV0::Kernel(
            CorrespondenceErrorV0::UnregisteredConfiguration
        ))
    );
    let mut incompatible = packet.clone();
    incompatible.population.runoff_policy = RunoffPolicyV0::UniformPerAreaV0 { rate: 0.2 };
    incompatible.population.population_definition_hash =
        population_definition_hash_v0(&incompatible.population).unwrap();
    incompatible.derived_common_packet_hash =
        landform_object_packet_hash_v0(&incompatible).unwrap();
    assert_eq!(
        validate_correspondence_pair_v0(&packet, &incompatible, CorrespondenceConfigV0::default()),
        Err(PacketCorrespondenceErrorV0::Incompatible(
            "common evaluation population"
        ))
    );
}

fn topology_assignments(
    source: &[AssignmentObjectInputV0],
    target: &[AssignmentObjectInputV0],
    scores: &[PositiveScoreV0],
) -> AssignmentKernelOutputV0 {
    build_assignment_kernel_v0(
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
        source,
        target,
        scores,
    )
    .unwrap()
}

#[test]
fn o0b_production_topology_reports_unique_none_and_tied_some_without_reranking() {
    let source = [
        assignment_object(0, 1.0, 0.0),
        assignment_object(1, 1.0, 1.0),
    ];
    let target = [
        assignment_object(10, 1.0, 10.0),
        assignment_object(11, 1.0, 11.0),
    ];
    let unique = topology_assignments(&source, &target, &[score(0, 10, 1.0), score(1, 11, 1.0)]);
    let none = build_topology_records_v0(
        PacketSideV0::Source,
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
        &[TopologyEdgeInputV0 {
            from_id: 0,
            target: TopologyTargetV0::DrainageNode(1),
            hierarchy_ambiguous: false,
        }],
        &[
            TopologyObjectInputV0 {
                object_id: 10,
                target: TopologyTargetV0::Portal(41),
            },
            TopologyObjectInputV0 {
                object_id: 11,
                target: TopologyTargetV0::DrainageNode(10),
            },
        ],
        &unique.assignments,
        &unique.best_components,
        &[41],
    )
    .unwrap();
    assert_eq!(none.len(), 1);
    assert_eq!(none[0].availability, TopologyAvailabilityV0::Available);
    assert_eq!(none[0].mapped_adjacency, Some(MappedAdjacencyV0::None));
    assert_eq!(none[0].endpoints_in_same_best_component, Some(false));

    let tied_target = [
        assignment_object(20, 1.0, 20.0),
        assignment_object(21, 1.0, 21.0),
        assignment_object(22, 1.0, 22.0),
    ];
    let tied = topology_assignments(
        &source,
        &tied_target,
        &[
            score(0, 20, 1.0),
            score(0, 21, 1.0),
            PositiveScoreV0 {
                source_id: 0,
                target_id: 22,
                source_score: 0.5,
                target_score: 1.0,
            },
            score(1, 22, 1.0),
        ],
    );
    let some = build_topology_records_v0(
        PacketSideV0::Source,
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
        &[TopologyEdgeInputV0 {
            from_id: 0,
            target: TopologyTargetV0::DrainageNode(1),
            hierarchy_ambiguous: false,
        }],
        &[
            TopologyObjectInputV0 {
                object_id: 20,
                target: TopologyTargetV0::DrainageNode(22),
            },
            TopologyObjectInputV0 {
                object_id: 21,
                target: TopologyTargetV0::Portal(41),
            },
            TopologyObjectInputV0 {
                object_id: 22,
                target: TopologyTargetV0::Portal(41),
            },
        ],
        &tied.assignments,
        &tied.best_components,
        &[41],
    )
    .unwrap();
    assert_eq!(some[0].mapped_adjacency, Some(MappedAdjacencyV0::Some));
    assert_eq!(some[0].endpoints_in_same_best_component, Some(true));
    assert_eq!(
        assignment(&tied, PacketSideV0::Source, 0).maximum_partner_ids,
        vec![20, 21]
    );
}

#[test]
fn o0b_production_topology_preserves_portal_root_and_unavailable_semantics() {
    let source = [assignment_object(0, 1.0, 0.0)];
    let target = [assignment_object(10, 1.0, 10.0)];
    let drainage = topology_assignments(&source, &target, &[score(0, 10, 1.0)]);
    let portal_edge = [TopologyEdgeInputV0 {
        from_id: 0,
        target: TopologyTargetV0::Portal(41),
        hierarchy_ambiguous: false,
    }];
    let portal_all = build_topology_records_v0(
        PacketSideV0::Source,
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
        &portal_edge,
        &[TopologyObjectInputV0 {
            object_id: 10,
            target: TopologyTargetV0::Portal(41),
        }],
        &drainage.assignments,
        &drainage.best_components,
        &[41],
    )
    .unwrap();
    assert_eq!(portal_all[0].mapped_adjacency, Some(MappedAdjacencyV0::All));
    assert_eq!(portal_all[0].endpoints_in_same_best_component, None);

    let portal_mismatch = build_topology_records_v0(
        PacketSideV0::Source,
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
        &portal_edge,
        &[TopologyObjectInputV0 {
            object_id: 10,
            target: TopologyTargetV0::Portal(42),
        }],
        &drainage.assignments,
        &drainage.best_components,
        &[41, 42],
    )
    .unwrap();
    assert_eq!(
        portal_mismatch[0].mapped_adjacency,
        Some(MappedAdjacencyV0::None)
    );
    assert_eq!(
        build_topology_records_v0(
            PacketSideV0::Source,
            ObjectFamilyV0::DrainageNode,
            AssignmentChannelV0::DrainageLine,
            &[TopologyEdgeInputV0 {
                from_id: 0,
                target: TopologyTargetV0::Portal(42),
                hierarchy_ambiguous: false,
            }],
            &[TopologyObjectInputV0 {
                object_id: 10,
                target: TopologyTargetV0::Portal(41),
            }],
            &drainage.assignments,
            &drainage.best_components,
            &[41],
        ),
        Err(CorrespondenceErrorV0::UndeclaredPortal(42))
    );

    let highland = build_assignment_kernel_v0(
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &source,
        &target,
        &[score(0, 10, 1.0)],
    )
    .unwrap();
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
        &highland.assignments,
        &highland.best_components,
        &[],
    )
    .unwrap();
    assert_eq!(root[0].mapped_adjacency, Some(MappedAdjacencyV0::All));
    assert_eq!(root[0].endpoints_in_same_best_component, None);

    let ambiguous = build_topology_records_v0(
        PacketSideV0::Source,
        ObjectFamilyV0::Highland,
        AssignmentChannelV0::HighlandExclusiveArea,
        &[TopologyEdgeInputV0 {
            from_id: 0,
            target: TopologyTargetV0::HighlandRoot,
            hierarchy_ambiguous: true,
        }],
        &[],
        &highland.assignments,
        &highland.best_components,
        &[],
    )
    .unwrap();
    assert_eq!(
        ambiguous[0].availability,
        TopologyAvailabilityV0::HierarchyAmbiguous
    );
    assert_eq!(ambiguous[0].mapped_adjacency, None);
    assert_eq!(ambiguous[0].endpoints_in_same_best_component, None);

    let null_assignment = topology_assignments(&source, &target, &[]);
    let missing = build_topology_records_v0(
        PacketSideV0::Source,
        ObjectFamilyV0::DrainageNode,
        AssignmentChannelV0::DrainageLine,
        &portal_edge,
        &[TopologyObjectInputV0 {
            object_id: 10,
            target: TopologyTargetV0::Portal(41),
        }],
        &null_assignment.assignments,
        &null_assignment.best_components,
        &[41],
    )
    .unwrap();
    assert_eq!(
        missing[0].availability,
        TopologyAvailabilityV0::NoMappedEndpoint
    );
    assert_eq!(missing[0].mapped_adjacency, None);
    assert_eq!(missing[0].endpoints_in_same_best_component, None);
}

fn asymmetric_y_reach_labels(
    packet: &LandformObjectPacketCoreV0,
) -> std::collections::BTreeMap<u32, &'static str> {
    let scale = packet
        .drainage
        .scales
        .iter()
        .find(|scale| scale.support_threshold_km2 == 2_000.0)
        .unwrap();
    let labels = scale
        .reach_graph
        .reaches
        .iter()
        .map(|reach| {
            let label = if reach.terminal_portal_id.is_some() {
                "trunk"
            } else {
                let head = packet.graph.cell_center_km[reach.cells[0] as usize];
                if head.x < 0.0 {
                    "west"
                } else {
                    "east"
                }
            };
            (reach.id, label)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        labels
            .values()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        { std::collections::BTreeSet::from(["trunk", "west", "east"]) }
    );
    labels
}

#[test]
#[ignore = "explicit O0b 8/4/2 correspondence evidence audit"]
fn o0b_asymmetric_y_4_to_8_and_2_keeps_unique_drainage_labels_in_both_channels() {
    let source = super::packet_tests::assembled_asymmetric_y_packet_at(4.0);
    let source_labels = asymmetric_y_reach_labels(&source);
    for target_spacing in [8.0, 2.0] {
        let target = super::packet_tests::assembled_asymmetric_y_packet_at(target_spacing);
        let target_labels = asymmetric_y_reach_labels(&target);
        let artifact = build_object_correspondence_v0(&source, &target).unwrap();
        eprintln!(
            "O0b asymmetric-Y 4->{target_spacing}: source_packet_bytes={} target_packet_bytes={} correspondence_bytes={} work={:?}",
            landform_object_packet_bytes_v0(&source).unwrap().len(),
            landform_object_packet_bytes_v0(&target).unwrap().len(),
            object_correspondence_bytes_v0(&artifact).unwrap().len(),
            artifact.work_counts,
        );
        assert!(artifact.metric_conflicts.is_empty());
        for row in artifact.assignment_records.iter().filter(|row| {
            row.family == ObjectFamilyV0::DrainageNode
                && matches!(
                    row.channel,
                    AssignmentChannelV0::DrainageExclusiveArea | AssignmentChannelV0::DrainageLine
                )
        }) {
            assert_eq!(row.support_status, SupportStatusV0::Eligible);
            assert_eq!(row.maximum_partner_ids.len(), 1);
            let partner = row.maximum_partner_ids[0];
            let (own_label, partner_label) = match row.side {
                PacketSideV0::Source => (source_labels[&row.object_id], target_labels[&partner]),
                PacketSideV0::Target => (target_labels[&row.object_id], source_labels[&partner]),
            };
            assert_eq!(own_label, partner_label);
            assert!(!row.exact_best_tie);
            assert!(row.normalized_margin.is_some_and(|margin| margin > 0.0));
        }
    }
}

fn linked_four_cone_labels(
    packet: &LandformObjectPacketCoreV0,
) -> std::collections::BTreeMap<u32, usize> {
    let cone_centers = [
        DVec3::new(-180.0, -40.0, 0.0),
        DVec3::new(-60.0, 0.0, 0.0),
        DVec3::new(60.0, 0.0, 0.0),
        DVec3::new(180.0, 40.0, 0.0),
    ];
    assert_eq!(packet.surface_hierarchy.populations.reference.len(), 4);
    let labels = packet
        .surface_hierarchy
        .populations
        .reference
        .iter()
        .map(|&peak_id| {
            let peak = &packet.surface_hierarchy.peaks[peak_id as usize];
            let anchor = packet.graph.cell_center_km[peak.anchor_cell as usize];
            let label = cone_centers
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    anchor
                        .distance_squared(**a)
                        .total_cmp(&anchor.distance_squared(**b))
                })
                .unwrap()
                .0;
            (peak_id, label)
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(
        labels
            .values()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([0, 1, 2, 3])
    );
    labels
}

fn assert_linked_four_cone_correspondence(target_spacing: f64) {
    let source = super::packet_tests::assembled_linked_four_cone_packet_at(4.0);
    let source_labels = linked_four_cone_labels(&source);
    let target = super::packet_tests::assembled_linked_four_cone_packet_at(target_spacing);
    let target_labels = linked_four_cone_labels(&target);
    let artifact = build_object_correspondence_v0(&source, &target).unwrap();
    eprintln!(
        "O0b linked-four-cone 4->{target_spacing}: source_packet_bytes={} target_packet_bytes={} correspondence_bytes={} work={:?}",
        landform_object_packet_bytes_v0(&source).unwrap().len(),
        landform_object_packet_bytes_v0(&target).unwrap().len(),
        object_correspondence_bytes_v0(&artifact).unwrap().len(),
        artifact.work_counts,
    );
    for row in artifact.assignment_records.iter().filter(|row| {
        row.family == ObjectFamilyV0::Highland
            && row.channel == AssignmentChannelV0::HighlandExclusiveArea
    }) {
        assert_eq!(row.support_status, SupportStatusV0::Eligible);
        assert_eq!(row.maximum_partner_ids.len(), 1);
        let partner = row.maximum_partner_ids[0];
        let (own_label, partner_label) = match row.side {
            PacketSideV0::Source => (source_labels[&row.object_id], target_labels[&partner]),
            PacketSideV0::Target => (target_labels[&row.object_id], source_labels[&partner]),
        };
        assert_eq!(own_label, partner_label);
        assert!(!row.exact_best_tie);
        assert!(row.normalized_margin.is_some_and(|margin| margin > 0.0));
    }
}

#[test]
#[ignore = "explicit O0b linked-four-cone 4-to-8 evidence audit"]
fn o0b_linked_four_cone_4_to_8_keeps_unique_highland_labels() {
    assert_linked_four_cone_correspondence(8.0);
}

#[test]
#[ignore = "frozen O0b gate currently halts at 2 km D0 depression-hierarchy ambiguity; see dated audit"]
fn o0b_linked_four_cone_4_to_2_keeps_unique_highland_labels() {
    assert_linked_four_cone_correspondence(2.0);
}

fn rectangle_graph(rectangles: &[Rect]) -> EvaluationSurfaceGraphV0 {
    let mut offsets = Vec::with_capacity(rectangles.len() + 1);
    let mut vertices = Vec::with_capacity(4 * rectangles.len());
    offsets.push(0);
    for rectangle in rectangles {
        vertices.extend(rectangle.polygon());
        offsets.push(vertices.len() as u32);
    }
    EvaluationSurfaceGraphV0 {
        domain: EvaluationDomainV0::Planar,
        cell_center_km: rectangles
            .iter()
            .map(|rectangle| {
                DVec3::new(
                    0.5 * (rectangle.x0 + rectangle.x1),
                    0.5 * (rectangle.y0 + rectangle.y1),
                    0.0,
                )
            })
            .collect(),
        cell_area_km2: rectangles
            .iter()
            .map(|rectangle| rectangle.area())
            .collect(),
        cell_polygon_offsets: offsets,
        cell_polygon_vertices_km: vertices,
        edge_offsets: vec![0; rectangles.len() + 1],
        edge_neighbor: Vec::new(),
        edge_reciprocal: Vec::new(),
        edge_distance_km: Vec::new(),
        edge_shared_width_km: Vec::new(),
        edge_face_endpoints_km: Vec::new(),
        boundary_segments: Vec::new(),
    }
}

fn one_object_population(
    family: ObjectFamilyV0,
    object_id: u32,
    object_cells: Vec<u32>,
    cell_class: Vec<PacketCellClassV0>,
) -> PacketAreaPopulationV0 {
    PacketAreaPopulationV0 {
        family,
        objects: vec![PacketAreaObjectV0 {
            id: object_id,
            status: SupportStatusV0::Eligible,
            nested_cells: object_cells.clone(),
            exclusive_cells: object_cells,
        }],
        cell_class,
    }
}

#[test]
fn o0b_production_area_population_kernel_retains_exact_background_and_portal_context() {
    let cells = [
        Rect::new(0.0, 10.0, 0.0, 10.0),
        Rect::new(10.0, 20.0, 0.0, 10.0),
        Rect::new(20.0, 30.0, 0.0, 10.0),
        Rect::new(30.0, 40.0, 0.0, 10.0),
    ];
    let graph = rectangle_graph(&cells);
    let source = one_object_population(
        ObjectFamilyV0::Highland,
        1,
        vec![0],
        vec![
            PacketCellClassV0::EligibleObject(1),
            PacketCellClassV0::HighlandBackground,
            PacketCellClassV0::HighlandBackground,
            PacketCellClassV0::HighlandBackground,
        ],
    );
    let target = one_object_population(
        ObjectFamilyV0::Highland,
        10,
        vec![2],
        vec![
            PacketCellClassV0::HighlandBackground,
            PacketCellClassV0::HighlandBackground,
            PacketCellClassV0::EligibleObject(10),
            PacketCellClassV0::HighlandBackground,
        ],
    );
    let background =
        build_area_population_kernel_v0(&graph, &source, &graph, &target, 1.0e-8, 1.0e-10).unwrap();
    assert!(background.exclusive_pairs.is_empty());
    assert!(background.nested_pairs.is_empty());
    assert_eq!(background.context_records.len(), 2);
    assert!(background.context_records.iter().all(|context| {
        context.background_area_km2 == 100.0
            && context.portal_areas_km2.is_empty()
            && context.outside_domain_area_km2 == 0.0
    }));

    let source_drainage = one_object_population(
        ObjectFamilyV0::DrainageNode,
        1,
        vec![0],
        vec![
            PacketCellClassV0::EligibleObject(1),
            PacketCellClassV0::Portal(7),
            PacketCellClassV0::Portal(7),
            PacketCellClassV0::Portal(7),
        ],
    );
    let target_drainage = one_object_population(
        ObjectFamilyV0::DrainageNode,
        10,
        vec![2],
        vec![
            PacketCellClassV0::Portal(7),
            PacketCellClassV0::Portal(7),
            PacketCellClassV0::EligibleObject(10),
            PacketCellClassV0::Portal(7),
        ],
    );
    let portal = build_area_population_kernel_v0(
        &graph,
        &source_drainage,
        &graph,
        &target_drainage,
        1.0e-8,
        1.0e-10,
    )
    .unwrap();
    assert!(portal.exclusive_pairs.is_empty());
    assert_eq!(portal.context_records.len(), 2);
    assert!(portal.context_records.iter().all(|context| {
        context.background_area_km2 == 0.0
            && context.portal_areas_km2.len() == 1
            && context.portal_areas_km2[0].portal_id == 7
            && context.portal_areas_km2[0].area_km2 == 100.0
            && context.outside_domain_area_km2 == 0.0
    }));

    let one_cell_graph = rectangle_graph(&[Rect::new(0.0, 10.0, 0.0, 10.0)]);
    let eligible = one_object_population(
        ObjectFamilyV0::Highland,
        1,
        vec![0],
        vec![PacketCellClassV0::EligibleObject(1)],
    );
    let ambiguous = PacketAreaPopulationV0 {
        family: ObjectFamilyV0::Highland,
        objects: vec![PacketAreaObjectV0 {
            id: 10,
            status: SupportStatusV0::HierarchyAmbiguousSupport,
            nested_cells: vec![0],
            exclusive_cells: vec![0],
        }],
        cell_class: vec![PacketCellClassV0::IneligibleHighland {
            peak_id: 10,
            status: SupportStatusV0::HierarchyAmbiguousSupport,
        }],
    };
    let ineligible = build_area_population_kernel_v0(
        &one_cell_graph,
        &eligible,
        &one_cell_graph,
        &ambiguous,
        1.0e-8,
        1.0e-10,
    )
    .unwrap();
    assert_eq!(ineligible.nested_pairs.len(), 1);
    assert_eq!(ineligible.nested_pairs[0].intersection_area_km2, 100.0);
    assert!(ineligible.exclusive_pairs.is_empty());
    assert_eq!(ineligible.context_records.len(), 1);
    assert_eq!(ineligible.context_records[0].side, PacketSideV0::Source);
    assert_eq!(
        ineligible.context_records[0]
            .ineligible_highland_areas
            .len(),
        1
    );
    assert_eq!(
        ineligible.context_records[0].ineligible_highland_areas[0],
        IneligibleHighlandAreaV0 {
            peak_id: 10,
            support_status: SupportStatusV0::HierarchyAmbiguousSupport,
            area_km2: 100.0,
        }
    );

    let shifted_graph = rectangle_graph(&[Rect::new(5.0, 15.0, 0.0, 10.0)]);
    let shifted_population = one_object_population(
        ObjectFamilyV0::Highland,
        2,
        vec![0],
        vec![PacketCellClassV0::EligibleObject(2)],
    );
    let partial_domain = build_area_population_kernel_v0(
        &one_cell_graph,
        &eligible,
        &shifted_graph,
        &shifted_population,
        1.0e-8,
        1.0e-10,
    )
    .unwrap();
    assert_eq!(partial_domain.exclusive_pairs.len(), 1);
    assert_eq!(
        partial_domain.exclusive_pairs[0].intersection_area_km2,
        50.0
    );
    assert_eq!(partial_domain.context_records.len(), 2);
    assert!(partial_domain
        .context_records
        .iter()
        .all(|context| context.outside_domain_area_km2 == 50.0));
}

#[test]
fn o0b_production_sparse_indexes_reject_separated_cartesian_work() {
    let source_rectangles = (0..100)
        .map(|index| {
            let x = (index % 10) as f64;
            let y = (index / 10) as f64;
            Rect::new(x, x + 1.0, y, y + 1.0)
        })
        .collect::<Vec<_>>();
    let target_rectangles = source_rectangles
        .iter()
        .map(|rectangle| {
            Rect::new(
                rectangle.x0 + 100.0,
                rectangle.x1 + 100.0,
                rectangle.y0 + 100.0,
                rectangle.y1 + 100.0,
            )
        })
        .collect::<Vec<_>>();
    let source_graph = rectangle_graph(&source_rectangles);
    let target_graph = rectangle_graph(&target_rectangles);
    let source_area = one_object_population(
        ObjectFamilyV0::Highland,
        1,
        (0..100).collect(),
        vec![PacketCellClassV0::EligibleObject(1); 100],
    );
    let target_area = one_object_population(
        ObjectFamilyV0::Highland,
        2,
        (0..100).collect(),
        vec![PacketCellClassV0::EligibleObject(2); 100],
    );
    let area = build_area_population_kernel_v0(
        &source_graph,
        &source_area,
        &target_graph,
        &target_area,
        1.0e-8,
        1.0e-10,
    )
    .unwrap();
    assert_eq!(area.cell_box_candidates, 0);
    assert_eq!(area.polygon_clips, 0);
    assert_eq!(area.positive_cell_intersections, 0);
    assert!(area.exclusive_pairs.is_empty());
    let area_reversed = build_area_population_kernel_v0(
        &target_graph,
        &target_area,
        &source_graph,
        &source_area,
        1.0e-8,
        1.0e-10,
    )
    .unwrap();
    assert_eq!(area_reversed.cell_box_candidates, 0);
    assert_eq!(area_reversed.polygon_clips, 0);
    assert_eq!(area_reversed.positive_cell_intersections, 0);
    assert!(area_reversed.exclusive_pairs.is_empty());

    let source_segments = (0..100)
        .map(|index| {
            let x = (index % 10) as f64;
            let y = (index / 10) as f64;
            line_segment(x, x + 1.0, y, 0.5)
        })
        .map(|segment| vec![segment])
        .collect::<Vec<_>>();
    let target_segments = source_segments
        .iter()
        .map(|segments| {
            vec![LineSegmentInputV0 {
                endpoints_km: segments[0]
                    .endpoints_km
                    .map(|point| point + DVec3::new(100.0, 100.0, 0.0)),
                ..segments[0]
            }]
        })
        .collect::<Vec<_>>();
    let source_lines = source_segments
        .iter()
        .enumerate()
        .map(|(id, segments)| LineObjectInputV0 {
            object_id: id as u32,
            segments,
        })
        .collect::<Vec<_>>();
    let target_lines = target_segments
        .iter()
        .enumerate()
        .map(|(id, segments)| LineObjectInputV0 {
            object_id: id as u32,
            segments,
        })
        .collect::<Vec<_>>();
    let line = build_line_population_v0(&source_lines, &target_lines).unwrap();
    assert_eq!(line.source_segments, 100);
    assert_eq!(line.target_segments, 100);
    assert_eq!(line.segment_box_candidates, 0);
    assert_eq!(line.segment_pair_tests, 0);
    assert_eq!(line.segment_index_node_visits, 100);
    assert!(line.pairs.is_empty());
    let line_reversed = build_line_population_v0(&target_lines, &source_lines).unwrap();
    assert_eq!(line_reversed.source_segments, 100);
    assert_eq!(line_reversed.target_segments, 100);
    assert_eq!(line_reversed.segment_box_candidates, 0);
    assert_eq!(line_reversed.segment_pair_tests, 0);
    assert_eq!(line_reversed.segment_index_node_visits, 100);
    assert!(line_reversed.pairs.is_empty());
}
