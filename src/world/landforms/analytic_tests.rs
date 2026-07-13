//! Frozen amendment-B planar analytic 8/4/2 km matrix.

use super::*;

const WIDTH: f64 = 1120.0;
const HEIGHT: f64 = 480.0 * 1.732_050_807_568_877_2;
const SPACINGS: [f64; 3] = [8.0, 4.0, 2.0];
const LINKED: [(DVec3, f64); 4] = [
    (DVec3::new(-180.0, -40.0, 0.0), 2.4),
    (DVec3::new(-60.0, 0.0, 0.0), 2.0),
    (DVec3::new(60.0, 0.0, 0.0), 2.2),
    (DVec3::new(180.0, 40.0, 0.0), 1.9),
];

#[derive(Clone, Copy)]
struct ObjectOracle {
    name: &'static str,
    area: f64,
    length: f64,
    width: f64,
}

struct ObjectError {
    name: &'static str,
    spacing: f64,
    area: f64,
    length: f64,
    width: f64,
}

struct ResolutionReport {
    spacing: f64,
    errors: Vec<ObjectError>,
}

fn build_graph(spacing: f64) -> EvaluationSurfaceGraphV0 {
    let cols = (WIDTH / spacing).round() as usize;
    let row_step = spacing * 3.0_f64.sqrt() * 0.5;
    let rows = (HEIGHT / row_step).round() as usize;
    let expected = match spacing as u32 {
        8 => (140, 120),
        4 => (280, 240),
        2 => (560, 480),
        _ => panic!("unregistered spacing {spacing}"),
    };
    assert_eq!((cols, rows), expected);
    let mesh = LandscapeMesh::uniform_planar_hex_with_portals(WIDTH, HEIGHT, spacing, &[]).unwrap();
    assert_eq!(mesh.cell_count(), cols * rows);
    let y0 = -0.5 * (rows - 1) as f64 * row_step;
    let even_x0 = -0.5 * (cols - 1) as f64 * spacing - 0.25 * spacing;
    let odd_x0 = -0.5 * (cols - 1) as f64 * spacing + 0.25 * spacing;
    assert_eq!(mesh.cell_center_km[0], DVec3::new(even_x0, y0, 0.0));
    assert_eq!(
        mesh.cell_center_km[cols],
        DVec3::new(odd_x0, y0 + row_step, 0.0)
    );
    let mean = mesh.cell_center_km.iter().copied().sum::<DVec3>() / mesh.cell_count() as f64;
    assert!(mean.length() <= 1.0e-11, "phase mean {mean:?}");
    let config = SurfaceHierarchyConfigV0::default();
    let controls = build_regular_hex_control_volumes_v0(&mesh, &config).unwrap();
    let graph = adapt_landscape_graph_v0(&mesh, &controls, &config).unwrap();
    graph.validate(&config).unwrap();
    for cell in 0..graph.cell_count() {
        let area = planar_polygon_signed_area(graph.polygon(cell));
        let relative = (area - graph.cell_area_km2[cell]).abs() / graph.cell_area_km2[cell];
        assert!(relative <= config.planar_area_match_relative);
    }
    graph
}

fn sample(graph: &EvaluationSurfaceGraphV0, f: impl Fn(DVec3) -> f64) -> Vec<f64> {
    graph.cell_center_km.iter().copied().map(f).collect()
}

fn extract(graph: &EvaluationSurfaceGraphV0, elevation: &[f64]) -> SurfaceHierarchyV0 {
    build_surface_hierarchy_v0(
        graph,
        elevation,
        &vec![true; graph.cell_count()],
        SurfaceHierarchyConfigV0::default(),
    )
    .unwrap()
}

fn measurements(h: &SurfaceHierarchyV0, peak: usize) -> &PlanarHighlandMeasurementsV0 {
    let feature = h
        .reference_highlands
        .iter()
        .find(|feature| feature.peak_id as usize == peak)
        .unwrap_or_else(|| panic!("peak {peak} not retained"));
    let HighlandMeasurementsV0::Planar(value) = &feature.measurements else {
        panic!("planar fixture returned non-planar measurements")
    };
    value
}

fn match_labels(
    graph: &EvaluationSurfaceGraphV0,
    h: &SurfaceHierarchyV0,
    labels: &[DVec3],
) -> Vec<usize> {
    let mut result = Vec::new();
    for &label in labels {
        let id = h
            .peaks
            .iter()
            .min_by(|a, b| {
                let da = graph.cell_center_km[a.anchor_cell as usize].distance_squared(label);
                let db = graph.cell_center_km[b.anchor_cell as usize].distance_squared(label);
                da.total_cmp(&db).then_with(|| a.id.cmp(&b.id))
            })
            .unwrap()
            .id as usize;
        assert!(!result.contains(&id));
        result.push(id);
    }
    result
}

fn assert_apex(
    graph: &EvaluationSurfaceGraphV0,
    peak: &PeakBranchV0,
    center: DVec3,
    height: f64,
    slope: f64,
    spacing: f64,
) {
    let circumradius = spacing / 3.0_f64.sqrt();
    let anchor = graph.cell_center_km[peak.anchor_cell as usize];
    assert!(anchor.distance(center) <= circumradius + 1.0e-10);
    assert!((peak.peak_elevation_km - height).abs() <= slope * circumradius + 1.0e-12);
}

fn polygon_distance(point: DVec3, polygon: &[DVec3]) -> f64 {
    let mut inside = true;
    let mut distance = f64::INFINITY;
    for (a, b) in polygon
        .iter()
        .copied()
        .zip(polygon.iter().copied().cycle().skip(1))
        .take(polygon.len())
    {
        let edge = b - a;
        let offset = point - a;
        inside &= edge.x * offset.y - edge.y * offset.x >= 0.0;
        distance = distance.min(point_segment_distance(point, [a, b]));
    }
    if inside {
        0.0
    } else {
        distance
    }
}

fn assert_contact(
    graph: &EvaluationSurfaceGraphV0,
    saddle: &SaddleNodeV0,
    elevation: &[f64],
    point: DVec3,
    height: f64,
    slope: f64,
    spacing: f64,
) {
    let sampling_radius = spacing + spacing / 3.0_f64.sqrt();
    assert!((saddle.elevation_km - height).abs() <= slope * sampling_radius + 1.0e-12);
    let mut support = saddle
        .flat_saddle_cells
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    for &cell in &saddle.flat_saddle_cells {
        for edge in edge_range(&graph.edge_offsets, cell as usize) {
            let neighbor = graph.edge_neighbor[edge] as usize;
            if elevation[neighbor] > saddle.elevation_km {
                support.insert(neighbor as u32);
            }
        }
    }
    let distance = support
        .iter()
        .map(|&cell| polygon_distance(point, graph.polygon(cell as usize)))
        .fold(f64::INFINITY, f64::min);
    assert!(
        distance <= sampling_radius + 1.0e-10,
        "support distance {distance} from contact {point:?} at saddle {}",
        saddle.elevation_km
    );
}

fn assert_buffer(graph: &EvaluationSurfaceGraphV0, h: &SurfaceHierarchyV0) {
    const R: f64 = 100.0;
    let mut buckets = BTreeMap::<(i64, i64), Vec<(u32, DVec3)>>::new();
    for &peak_id in &h.populations.reference {
        let peak = &h.peaks[peak_id as usize];
        for &cell in &peak.footprint_members {
            let p = graph.cell_center_km[cell as usize];
            buckets
                .entry(((p.x / R).floor() as i64, (p.y / R).floor() as i64))
                .or_default()
                .push((cell, p));
        }
        let relief = measurements(h, peak_id as usize)
            .local_relief
            .iter()
            .find(|x| x.radius_km == R)
            .unwrap();
        assert!(relief.truncated_member_cells.is_empty());
    }
    for segment in &graph.boundary_segments {
        let min_x = segment.endpoints_km[0].x.min(segment.endpoints_km[1].x) - R;
        let max_x = segment.endpoints_km[0].x.max(segment.endpoints_km[1].x) + R;
        let min_y = segment.endpoints_km[0].y.min(segment.endpoints_km[1].y) - R;
        let max_y = segment.endpoints_km[0].y.max(segment.endpoints_km[1].y) + R;
        for bx in (min_x / R).floor() as i64..=(max_x / R).floor() as i64 {
            for by in (min_y / R).floor() as i64..=(max_y / R).floor() as i64 {
                if let Some(cells) = buckets.get(&(bx, by)) {
                    for &(cell, p) in cells {
                        let distance = point_segment_distance(p, segment.endpoints_km);
                        assert!(distance > R, "cell {cell} boundary distance {distance}");
                    }
                }
            }
        }
    }
}

fn assert_area_closure(h: &SurfaceHierarchyV0) {
    let tolerance = SurfaceHierarchyConfigV0::default().planar_area_match_relative;
    for &id in &h.populations.reference {
        let peak = &h.peaks[id as usize];
        let area = measurements(h, id as usize).footprint_geometry.area_km2;
        assert!((area - peak.footprint_area_km2).abs() / peak.footprint_area_km2 <= tolerance);
    }
}

fn error(spacing: f64, oracle: ObjectOracle, g: &PlanarFootprintGeometryV0) -> ObjectError {
    ObjectError {
        name: oracle.name,
        spacing,
        area: (g.area_km2 - oracle.area).abs() / oracle.area,
        length: (g.equivalent_ellipse_length_km - oracle.length).abs() / oracle.length,
        width: (g.equivalent_ellipse_width_km - oracle.width).abs() / oracle.width,
    }
}

fn contact(a: (DVec3, f64), b: (DVec3, f64)) -> (DVec3, f64) {
    let slope = 0.010;
    let distance = a.0.distance(b.0);
    let along = (a.1 - b.1 + slope * distance) / (2.0 * slope);
    (a.0 + (b.0 - a.0) * (along / distance), a.1 - slope * along)
}

fn run_resolution(spacing: f64) -> ResolutionReport {
    let graph = build_graph(spacing);
    let mut errors = Vec::new();

    let z = sample(&graph, |p| 2.0 - 0.010 * p.length());
    let one = extract(&graph, &z);
    assert_eq!(
        (one.peaks.len(), one.saddles.len(), one.roots.len()),
        (1, 0, 1)
    );
    assert_eq!(one.populations.reference, vec![0]);
    assert_apex(&graph, &one.peaks[0], DVec3::ZERO, 2.0, 0.010, spacing);
    assert!(
        measurements(&one, 0)
            .footprint_geometry
            .orientation_ambiguous
    );
    assert_buffer(&graph, &one);
    assert_area_closure(&one);

    let two_centers = [DVec3::new(-80.0, 0.0, 0.0), DVec3::new(80.0, 0.0, 0.0)];
    let z = sample(&graph, |p| {
        (2.0 - 0.010 * p.distance(two_centers[0])).max(1.8 - 0.010 * p.distance(two_centers[1]))
    });
    let two = extract(&graph, &z);
    assert_eq!(
        (two.peaks.len(), two.saddles.len(), two.roots.len()),
        (2, 1, 1)
    );
    assert_eq!(two.populations.reference.len(), 2);
    let ids = match_labels(&graph, &two, &two_centers);
    assert_eq!(two.roots, vec![ids[0] as u32]);
    assert_eq!(two.peaks[ids[1]].parent_peak, Some(ids[0] as u32));
    assert_apex(
        &graph,
        &two.peaks[ids[0]],
        two_centers[0],
        2.0,
        0.010,
        spacing,
    );
    assert_apex(
        &graph,
        &two.peaks[ids[1]],
        two_centers[1],
        1.8,
        0.010,
        spacing,
    );
    assert_contact(
        &graph,
        &two.saddles[0],
        &z,
        DVec3::new(10.0, 0.0, 0.0),
        1.10,
        0.010,
        spacing,
    );
    assert_buffer(&graph, &two);
    assert_area_closure(&two);

    let narrow_z = sample(&graph, |p| 2.0 - 0.015 * p.length());
    let broad_z = sample(&graph, |p| 2.0 - 0.015 * (p.length() - 40.0).max(0.0));
    let narrow = extract(&graph, &narrow_z);
    let broad = extract(&graph, &broad_z);
    for h in [&narrow, &broad] {
        assert_eq!((h.peaks.len(), h.saddles.len(), h.roots.len()), (1, 0, 1));
        assert_eq!(h.populations.reference, vec![0]);
        assert!(measurements(h, 0).footprint_geometry.orientation_ambiguous);
        assert_buffer(&graph, h);
        assert_area_closure(h);
    }
    assert_apex(&graph, &narrow.peaks[0], DVec3::ZERO, 2.0, 0.015, spacing);
    let expected_support = graph
        .cell_center_km
        .iter()
        .enumerate()
        .filter_map(|(cell, p)| (p.length() <= 40.0).then_some(cell as u32))
        .collect::<Vec<_>>();
    assert!(!expected_support.is_empty());
    assert_eq!(broad.peaks[0].flat_maximum_cells, expected_support);
    let narrow_caps = &measurements(&narrow, 0).summit_caps;
    let broad_caps = &measurements(&broad, 0).summit_caps;
    assert_eq!(narrow_caps.len(), 3);
    assert_eq!(broad_caps.len(), 3);
    for (n, b) in narrow_caps.iter().zip(broad_caps) {
        assert!(b.area_km2 > n.area_km2);
        assert_eq!(n.valid_grade_fraction, 1.0);
        assert_eq!(b.valid_grade_fraction, 1.0);
        assert!(!n.cap_merge_censored);
        assert!(!b.cap_merge_censored);
        assert!(b.gentle_fractions[0].fraction > n.gentle_fractions[0].fraction);
        assert!(b.gentle_fractions[1].fraction > n.gentle_fractions[1].fraction);
        assert!(b.gentle_fractions[2].fraction >= n.gentle_fractions[2].fraction);
    }

    let z = sample(&graph, |p| {
        LINKED
            .iter()
            .map(|&(center, height)| height - 0.010 * p.distance(center))
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let linked = extract(&graph, &z);
    assert_eq!(
        (linked.peaks.len(), linked.saddles.len(), linked.roots.len()),
        (4, 3, 1)
    );
    assert_eq!(linked.populations.reference.len(), 4);
    let labels = LINKED.map(|cone| cone.0);
    let ids = match_labels(&graph, &linked, &labels);
    assert_eq!(linked.roots, vec![ids[0] as u32]);
    for &loser in &ids[1..] {
        assert_eq!(linked.peaks[loser].parent_peak, Some(ids[0] as u32));
    }
    for (&id, &(center, height)) in ids.iter().zip(&LINKED) {
        assert_apex(&graph, &linked.peaks[id], center, height, 0.010, spacing);
    }
    let pairs = [(0usize, 1usize), (1, 2), (2, 3)];
    let losers = [ids[1], ids[2], ids[3]];
    let mut previous = f64::INFINITY;
    for ((a, b), loser) in pairs.into_iter().zip(losers) {
        let (point, height) = contact(LINKED[a], LINKED[b]);
        let saddle = &linked.saddles[linked.peaks[loser].key_saddle.unwrap() as usize];
        assert!(saddle.elevation_km < previous);
        previous = saddle.elevation_km;
        assert_eq!(saddle.losing_peaks, vec![loser as u32]);
        assert_contact(&graph, saddle, &z, point, height, 0.010, spacing);
    }
    assert_buffer(&graph, &linked);
    assert_area_closure(&linked);
    let linked_oracles = [
        ObjectOracle {
            name: "linked-root-A",
            area: 307_636.562_39,
            length: 831.541_290,
            width: 477.053_652,
        },
        ObjectOracle {
            name: "linked-loser-B",
            area: 5_875.337_063,
            length: 86.491_106_4,
            width: 86.491_106_4,
        },
        ObjectOracle {
            name: "linked-loser-C",
            area: 15_393.804_003,
            length: 140.0,
            width: 140.0,
        },
        ObjectOracle {
            name: "linked-loser-D",
            area: 7_312.476_002,
            length: 96.491_106_4,
            width: 96.491_106_4,
        },
    ];
    for (&id, oracle) in ids.iter().zip(linked_oracles) {
        errors.push(error(
            spacing,
            oracle,
            &measurements(&linked, id).footprint_geometry,
        ));
    }

    let rectangle = ObjectOracle {
        name: "rectangle-axis",
        area: 38_400.0,
        length: 369.504_172_281,
        width: 138.564_064_606,
    };
    for (name, angle) in [
        ("rectangle-axis", 0.0_f64),
        ("rectangle-rotated-30", std::f64::consts::PI / 6.0),
    ] {
        let (sin, cos) = angle.sin_cos();
        let z = sample(&graph, |p| {
            let x = cos * p.x + sin * p.y;
            let y = -sin * p.x + cos * p.y;
            if x.abs() <= 160.0 && y.abs() <= 60.0 {
                1.0
            } else {
                0.0
            }
        });
        let h = extract(&graph, &z);
        assert_eq!((h.peaks.len(), h.saddles.len(), h.roots.len()), (1, 0, 1));
        assert_eq!(h.populations.reference, vec![0]);
        assert_buffer(&graph, &h);
        assert_area_closure(&h);
        let mut oracle = rectangle;
        oracle.name = name;
        errors.push(error(
            spacing,
            oracle,
            &measurements(&h, 0).footprint_geometry,
        ));
    }
    ResolutionReport { spacing, errors }
}

#[test]
#[ignore = "full registered 8/4/2 analytic landform matrix is an audit test"]
fn frozen_planar_analytic_8_4_2_matrix() {
    let reports = SPACINGS.map(run_resolution);
    for report in &reports {
        for error in &report.errors {
            println!(
                "{} km {}: area={:.6}%, length={:.6}%, width={:.6}%",
                error.spacing,
                error.name,
                100.0 * error.area,
                100.0 * error.length,
                100.0 * error.width
            );
            assert_eq!(error.spacing, report.spacing);
        }
    }
    for error in &reports[2].errors {
        assert!(
            error.area <= 0.05,
            "{} area error {}",
            error.name,
            error.area
        );
        assert!(
            error.length <= 0.05,
            "{} length error {}",
            error.name,
            error.length
        );
        assert!(
            error.width <= 0.075,
            "{} width error {}",
            error.name,
            error.width
        );
    }
}
