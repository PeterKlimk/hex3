use super::*;
use crate::geometry::fibonacci_sphere_points_with_rng;
use glam::{DQuat, DVec2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::OnceLock;

const VALIDATION_RADIUS_KM: f64 = 100.0;

fn validation_sphere() -> &'static EvaluationSurfaceGraphV0 {
    static GRAPH: OnceLock<EvaluationSurfaceGraphV0> = OnceLock::new();
    GRAPH.get_or_init(|| {
        let mut rng = ChaCha8Rng::seed_from_u64(0x0005_3050_0001);
        let points = fibonacci_sphere_points_with_rng(2_048, 0.0, &mut rng);
        let tessellation = Tessellation::from_points_knn_clipping(points);
        let config = SurfaceHierarchyConfigV0::default();
        let mut graph = adapt_product_tessellation_graph_v0(&tessellation, &config).unwrap();
        let scale = VALIDATION_RADIUS_KM / f64::from(PLANET_RADIUS_KM);
        graph.domain = EvaluationDomainV0::Spherical {
            radius_km: VALIDATION_RADIUS_KM,
        };
        for point in graph
            .cell_center_km
            .iter_mut()
            .chain(&mut graph.cell_polygon_vertices_km)
            .chain(graph.edge_face_endpoints_km.iter_mut().flatten())
        {
            *point = canonical_point(*point * scale);
        }
        for area in &mut graph.cell_area_km2 {
            *area *= scale * scale;
        }
        for distance in &mut graph.edge_distance_km {
            *distance *= scale;
        }
        for width in &mut graph.edge_shared_width_km {
            *width *= scale;
        }
        graph.validate(&config).unwrap();
        graph
    })
}

fn nearest_positive_z(graph: &EvaluationSurfaceGraphV0) -> usize {
    graph
        .cell_center_km
        .iter()
        .enumerate()
        .max_by(|(a_id, a), (b_id, b)| a.z.total_cmp(&b.z).then_with(|| b_id.cmp(a_id)))
        .unwrap()
        .0
}

fn exponential_map(center: DVec3, basis: [DVec3; 2], point: DVec2, radius: f64) -> DVec3 {
    let length = point.length();
    if length == 0.0 {
        return radius * center.normalize();
    }
    let tangent = (point.x * basis[0] + point.y * basis[1]) / length;
    radius * ((length / radius).cos() * center.normalize() + (length / radius).sin() * tangent)
}

fn geometry_only_graph(
    radius: f64,
    centers: Vec<DVec3>,
    polygons: Vec<Vec<DVec3>>,
    areas: Vec<f64>,
) -> EvaluationSurfaceGraphV0 {
    let cell_count = centers.len();
    let mut offsets = Vec::with_capacity(polygons.len() + 1);
    let mut vertices = Vec::new();
    for mut polygon in polygons {
        rotate_polygon_to_canonical_start(&mut polygon);
        offsets.push(vertices.len() as u32);
        vertices.extend(polygon.into_iter().map(canonical_point));
    }
    offsets.push(vertices.len() as u32);
    EvaluationSurfaceGraphV0 {
        domain: EvaluationDomainV0::Spherical { radius_km: radius },
        cell_center_km: centers.into_iter().map(canonical_point).collect(),
        cell_area_km2: areas,
        cell_polygon_offsets: offsets,
        cell_polygon_vertices_km: vertices,
        edge_offsets: vec![0; cell_count + 1],
        edge_neighbor: Vec::new(),
        edge_reciprocal: Vec::new(),
        edge_distance_km: Vec::new(),
        edge_shared_width_km: Vec::new(),
        edge_face_endpoints_km: Vec::new(),
        boundary_segments: Vec::new(),
    }
}

#[test]
fn manufactured_spherical_cap_builds_complete_repeatable_s0() {
    let graph = validation_sphere();
    let config = SurfaceHierarchyConfigV0::default();
    let center_cell = nearest_positive_z(graph);
    let center = graph.cell_center_km[center_cell];
    let elevation = graph
        .cell_center_km
        .iter()
        .map(|&point| {
            0.40 - 0.010 * VALIDATION_RADIUS_KM * spherical_angle_rad(center, point).unwrap()
        })
        .collect::<Vec<_>>();
    let scored = vec![true; graph.cell_count()];

    let first = build_surface_hierarchy_v0(graph, &elevation, &scored, config).unwrap();
    let second = build_surface_hierarchy_v0(graph, &elevation, &scored, config).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.derived_evidence_hash, second.derived_evidence_hash);
    assert_eq!(
        canonical_hierarchy_bytes(graph, &elevation, &scored, &config, &first).unwrap(),
        canonical_hierarchy_bytes(graph, &elevation, &scored, &config, &second).unwrap()
    );
    assert_eq!(first.roots.len(), 1);
    assert_eq!(first.peaks.len(), 1);
    assert_eq!(first.populations.reference, vec![0]);
    assert_eq!(first.reference_highlands.len(), 1);
    let HighlandMeasurementsV0::Spherical(measurements) =
        &first.reference_highlands[0].measurements
    else {
        panic!("manufactured sphere must produce spherical measurements");
    };
    let SphericalFootprintGeometryV0::Local(local) = &measurements.footprint_geometry else {
        panic!("the bounded cap must have local footprint geometry");
    };
    assert!(!local.spherical_nonlocal_warning);
    assert!(local.area_km2 >= config.reference_min_footprint_km2);
    assert!(local.equivalent_ellipse_length_km.is_finite());
    assert!(local.equivalent_ellipse_width_km.is_finite());
    assert!(measurements.two_sweep_extent_km.is_some());
    assert_eq!(measurements.local_relief.len(), 3);
    assert!(measurements
        .local_relief
        .iter()
        .all(|summary| summary.truncated_member_cells.is_empty()));
    assert_eq!(measurements.summit_caps.len(), 3);
}

#[test]
fn spherical_log_grade_and_relief_use_physical_arcs() {
    let graph = validation_sphere();
    let config = SurfaceHierarchyConfigV0::default();
    let center_cell = nearest_positive_z(graph);
    let center = graph.cell_center_km[center_cell];
    let basis = spherical_tangent_basis(center).unwrap();
    let mut scored = vec![false; graph.cell_count()];
    scored[center_cell] = true;
    for edge in edge_range(&graph.edge_offsets, center_cell) {
        scored[graph.edge_neighbor[edge] as usize] = true;
    }
    let elevation = graph
        .cell_center_km
        .iter()
        .map(|&point| {
            let q = spherical_log_xy(
                center,
                point,
                VALIDATION_RADIUS_KM,
                basis,
                config.linear_rank_relative,
            )
            .unwrap()
            .unwrap();
            0.010 * q.x
        })
        .collect::<Vec<_>>();
    let grades = physical_grades(graph, &elevation, &scored, &config).unwrap();
    assert!((grades[center_cell].unwrap() - 0.010).abs() <= 1.0e-12);

    let rotation = DQuat::from_axis_angle(DVec3::new(2.0, -1.0, 3.0).normalize(), 0.67);
    let mut rotated_graph = graph.clone();
    for point in &mut rotated_graph.cell_center_km {
        *point = rotation * *point;
    }
    let rotated_center = rotated_graph.cell_center_km[center_cell];
    let rotated_basis = spherical_tangent_basis(rotated_center).unwrap();
    let rotated_elevation = rotated_graph
        .cell_center_km
        .iter()
        .map(|&point| {
            let q = spherical_log_xy(
                rotated_center,
                point,
                VALIDATION_RADIUS_KM,
                rotated_basis,
                config.linear_rank_relative,
            )
            .unwrap()
            .unwrap();
            0.010 * q.x
        })
        .collect::<Vec<_>>();
    let rotated_grades =
        physical_grades(&rotated_graph, &rotated_elevation, &scored, &config).unwrap();
    assert!((rotated_grades[center_cell].unwrap() - 0.010).abs() <= 1.0e-12);

    let distances = [0.0, 24.0, 25.0, 26.0, 49.0, 50.0, 51.0, 99.0, 100.0, 101.0];
    let centers = distances
        .iter()
        .map(|distance| {
            let angle = distance / VALIDATION_RADIUS_KM;
            VALIDATION_RADIUS_KM * DVec3::new(angle.sin(), 0.0, angle.cos())
        })
        .collect::<Vec<_>>();
    let relief_graph = EvaluationSurfaceGraphV0 {
        domain: EvaluationDomainV0::Spherical {
            radius_km: VALIDATION_RADIUS_KM,
        },
        cell_center_km: centers,
        cell_area_km2: vec![1.0; distances.len()],
        cell_polygon_offsets: vec![0; distances.len() + 1],
        cell_polygon_vertices_km: Vec::new(),
        edge_offsets: vec![0; distances.len() + 1],
        edge_neighbor: Vec::new(),
        edge_reciprocal: Vec::new(),
        edge_distance_km: Vec::new(),
        edge_shared_width_km: Vec::new(),
        edge_face_endpoints_km: Vec::new(),
        boundary_segments: Vec::new(),
    };
    let fields = spherical_relief_fields(
        &relief_graph,
        &distances,
        &vec![true; distances.len()],
        &config,
        &[0],
    )
    .unwrap();
    for (index, expected) in [25.0, 50.0, 100.0].into_iter().enumerate() {
        assert_eq!(fields.relief_by_radius[index][0], Some(expected));
        assert!(!fields.truncated_by_radius[index][0]);
    }
}

#[test]
fn spherical_moments_are_rotation_covariant_and_nonlocal_is_reportable() {
    let radius = VALIDATION_RADIUS_KM;
    let center = DVec3::Z;
    let basis = spherical_tangent_basis(center).unwrap();
    let angle = 30.0_f64.to_radians();
    let rotate_2d = |point: DVec2| {
        DVec2::new(
            angle.cos() * point.x - angle.sin() * point.y,
            angle.sin() * point.x + angle.cos() * point.y,
        )
    };
    let rectangle_xy = [
        DVec2::new(-35.0, -12.0),
        DVec2::new(35.0, -12.0),
        DVec2::new(35.0, 12.0),
        DVec2::new(-35.0, 12.0),
    ]
    .map(rotate_2d);
    let rectangle = rectangle_xy.map(|point| exponential_map(center, basis, point, radius));
    let area = spherical_cell_area_km2(radius * center, &rectangle, radius).unwrap();
    let graph = geometry_only_graph(
        radius,
        vec![radius * center],
        vec![rectangle.to_vec()],
        vec![area],
    );
    let (geometry, _, _) =
        measure_spherical_footprint(&graph, &[0], &SurfaceHierarchyConfigV0::default(), 0).unwrap();
    let SphericalFootprintGeometryV0::Local(local) = geometry else {
        panic!("bounded rectangle must be local");
    };
    let expected_axis = angle.cos() * basis[0] + angle.sin() * basis[1];
    assert!(local.principal_axis.dot(expected_axis).abs() >= 1.0 - 1.0e-12);
    assert!(!local.orientation_ambiguous);

    let rotation = DQuat::from_axis_angle(DVec3::X.normalize(), 0.71);
    let rotated_polygon = rectangle.map(|point| rotation * point);
    let rotated_graph = geometry_only_graph(
        radius,
        vec![radius * (rotation * center)],
        vec![rotated_polygon.to_vec()],
        vec![area],
    );
    let (rotated_geometry, _, _) = measure_spherical_footprint(
        &rotated_graph,
        &[0],
        &SurfaceHierarchyConfigV0::default(),
        0,
    )
    .unwrap();
    let SphericalFootprintGeometryV0::Local(rotated) = rotated_geometry else {
        panic!("rigid rotation must remain local");
    };
    assert!(
        (rotated.equivalent_ellipse_length_km - local.equivalent_ellipse_length_km).abs() < 1.0e-10
    );
    assert!(
        (rotated.equivalent_ellipse_width_km - local.equivalent_ellipse_width_km).abs() < 1.0e-10
    );
    assert!(
        rotated
            .principal_axis
            .dot(rotation * local.principal_axis)
            .abs()
            >= 1.0 - 1.0e-12
    );

    let ring = (0..12)
        .map(|index| {
            let phi = std::f64::consts::TAU * index as f64 / 12.0;
            exponential_map(
                center,
                basis,
                20.0 * DVec2::new(phi.cos(), phi.sin()),
                radius,
            )
        })
        .collect::<Vec<_>>();
    let ring_area = spherical_cell_area_km2(radius * center, &ring, radius).unwrap();
    let ring_graph =
        geometry_only_graph(radius, vec![radius * center], vec![ring], vec![ring_area]);
    let (ring_geometry, _, _) =
        measure_spherical_footprint(&ring_graph, &[0], &SurfaceHierarchyConfigV0::default(), 0)
            .unwrap();
    let SphericalFootprintGeometryV0::Local(ring_local) = ring_geometry else {
        panic!("local symmetric ring must be measurable");
    };
    assert!(ring_local.orientation_ambiguous);

    let north = ring_graph.polygon(0).to_vec();
    let south = north.iter().rev().map(|point| -*point).collect::<Vec<_>>();
    let nonlocal_graph = geometry_only_graph(
        radius,
        vec![radius * DVec3::Z, -radius * DVec3::Z],
        vec![north, south],
        vec![ring_area, ring_area],
    );
    let (nonlocal, extent, mean_width) = measure_spherical_footprint(
        &nonlocal_graph,
        &[0, 1],
        &SurfaceHierarchyConfigV0::default(),
        0,
    )
    .unwrap();
    assert_eq!(nonlocal, SphericalFootprintGeometryV0::NonLocalGeometry);
    assert_eq!(extent, Some(std::f64::consts::PI * radius));
    assert!(mean_width.is_some());

    let peak = PeakBranchV0 {
        id: 0,
        peak_elevation_km: 1.0,
        anchor_cell: 0,
        flat_centroid_km: DVec3::ZERO,
        flat_maximum_cells: vec![0, 1],
        parent_peak: None,
        key_saddle: None,
        persistence_km: 1.0,
        root_closure: true,
        equal_elder_ambiguous: false,
        exclusive_cells: vec![0, 1],
        footprint_members: vec![0, 1],
        footprint_area_km2: 2.0 * ring_area,
        union_boundary_edges: Vec::new(),
        physical_boundary_segments: Vec::new(),
        scored_boundary_contact: false,
    };
    let features = build_reference_highlands(
        &nonlocal_graph,
        &[1.0, 1.0],
        &[true, true],
        &SurfaceHierarchyConfigV0::default(),
        &[peak],
        &[0],
    )
    .unwrap();
    let HighlandMeasurementsV0::Spherical(measurements) = &features[0].measurements else {
        panic!("nonlocal spherical geometry must retain spherical evidence");
    };
    assert_eq!(
        measurements.footprint_geometry,
        SphericalFootprintGeometryV0::NonLocalGeometry
    );
    assert_eq!(measurements.local_relief.len(), 3);
    assert_eq!(measurements.rank_deficient_grade_cells, vec![0, 1]);
    assert_eq!(measurements.summit_caps.len(), 3);
}

#[test]
fn spherical_boundary_arc_distance_uses_arc_interior_and_is_rotation_invariant() {
    let radius = VALIDATION_RADIUS_KM;
    let equator = |longitude: f64| radius * DVec3::new(longitude.cos(), longitude.sin(), 0.0);
    let endpoints = [equator(-0.2), equator(0.2)];
    let interior_query = radius * DVec3::new(0.1_f64.cos(), 0.0, 0.1_f64.sin());
    let endpoint_query = equator(0.3);
    let interior =
        spherical_point_minor_arc_distance_km(interior_query, endpoints, radius).unwrap();
    let endpoint =
        spherical_point_minor_arc_distance_km(endpoint_query, endpoints, radius).unwrap();
    assert!((interior - 10.0).abs() <= 1.0e-12);
    assert!((endpoint - 10.0).abs() <= 1.0e-12);

    let rotation = DQuat::from_axis_angle(DVec3::new(1.0, 2.0, 3.0).normalize(), 0.83);
    let rotated = spherical_point_minor_arc_distance_km(
        rotation * interior_query,
        endpoints.map(|point| rotation * point),
        radius,
    )
    .unwrap();
    assert!((rotated - interior).abs() <= 1.0e-12);
    let rotated_endpoint = spherical_point_minor_arc_distance_km(
        rotation * endpoint_query,
        endpoints.map(|point| rotation * point),
        radius,
    )
    .unwrap();
    assert!((rotated_endpoint - endpoint).abs() <= 1.0e-12);
}

#[test]
fn spherical_bucketed_relief_matches_exact_scan_and_boundary_arcs() {
    let graph = validation_sphere();
    let config = SurfaceHierarchyConfigV0::default();
    let scored = graph
        .cell_center_km
        .iter()
        .map(|center| center.x >= 0.0)
        .collect::<Vec<_>>();
    let elevation = graph
        .cell_center_km
        .iter()
        .map(|center| 0.003 * center.y + 0.002 * center.z)
        .collect::<Vec<_>>();
    let mut queries = (0..graph.cell_count())
        .filter(|&cell| scored[cell])
        .collect::<Vec<_>>();
    queries.sort_by(|&a, &b| {
        graph.cell_center_km[a]
            .x
            .abs()
            .total_cmp(&graph.cell_center_km[b].x.abs())
            .then_with(|| a.cmp(&b))
    });
    let queries = queries
        .into_iter()
        .take(6)
        .map(|cell| cell as u32)
        .collect::<Vec<_>>();
    let fields = spherical_relief_fields(graph, &elevation, &scored, &config, &queries).unwrap();
    let boundaries = (0..graph.cell_count())
        .flat_map(|cell| {
            let scored = &scored;
            edge_range(&graph.edge_offsets, cell).filter_map(move |edge| {
                let neighbor = graph.edge_neighbor[edge] as usize;
                (scored[cell] && !scored[neighbor]).then_some(graph.edge_face_endpoints_km[edge])
            })
        })
        .collect::<Vec<_>>();
    for &query in &queries {
        let query = query as usize;
        let center = graph.cell_center_km[query];
        let boundary_distance = boundaries
            .iter()
            .map(|&endpoints| {
                spherical_point_minor_arc_distance_km(center, endpoints, VALIDATION_RADIUS_KM)
                    .unwrap()
            })
            .fold(f64::INFINITY, f64::min);
        for (radius_index, &radius) in config.local_relief_radii_km.iter().enumerate() {
            let mut minimum = f64::INFINITY;
            let mut maximum = f64::NEG_INFINITY;
            for candidate in 0..graph.cell_count() {
                if !scored[candidate] {
                    continue;
                }
                let distance = VALIDATION_RADIUS_KM
                    * spherical_angle_rad(center, graph.cell_center_km[candidate]).unwrap();
                if distance <= radius {
                    minimum = minimum.min(elevation[candidate]);
                    maximum = maximum.max(elevation[candidate]);
                }
            }
            assert_eq!(
                fields.relief_by_radius[radius_index][query],
                Some(canonical_zero(maximum - minimum))
            );
            assert_eq!(
                fields.truncated_by_radius[radius_index][query],
                boundary_distance <= radius
            );
        }
    }
}
