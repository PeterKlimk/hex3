//! Manufactured D0 public-API gates.

use super::*;
use crate::world::landscape::{BoundarySide, OutletPortal};

fn graph_with_portals(
    width: f64,
    height: f64,
    spacing: f64,
    portals: &[OutletPortal],
) -> EvaluationSurfaceGraphV0 {
    let mesh =
        LandscapeMesh::uniform_planar_hex_with_portals(width, height, spacing, portals).unwrap();
    let config = SurfaceHierarchyConfigV0::default();
    let controls = build_regular_hex_control_volumes_v0(&mesh, &config).unwrap();
    adapt_landscape_graph_v0(&mesh, &controls, &config).unwrap()
}

fn south_portal(id: u32, width: f64, base_level_km: f32) -> OutletPortal {
    OutletPortal {
        id: OutletPortalId(id),
        side: BoundarySide::South,
        span_start_km: -0.5 * width,
        span_end_km: 0.5 * width,
        base_level_km,
    }
}

fn plane(graph: &EvaluationSurfaceGraphV0, offset: f64) -> Vec<f64> {
    graph
        .cell_center_km
        .iter()
        .map(|center| center.y + offset)
        .collect()
}

fn area_supply(graph: &EvaluationSurfaceGraphV0, rate: f64) -> Vec<f64> {
    graph.cell_area_km2.iter().map(|area| rate * area).collect()
}

fn segment_network_cost(point: DVec3, start: DVec3, end: DVec3, offset: f64) -> f64 {
    let segment = end - start;
    let length = segment.length();
    let t = ((point - start).dot(segment) / segment.length_squared()).clamp(0.0, 1.0);
    let projected = start + t * segment;
    offset + t * length + 2.0 * point.distance(projected)
}

fn y_valley_surface(graph: &EvaluationSurfaceGraphV0) -> Vec<f64> {
    let outlet = DVec3::new(0.0, -48.0, 0.0);
    let junction = DVec3::ZERO;
    let left_head = DVec3::new(-48.0, 48.0, 0.0);
    let right_head = DVec3::new(48.0, 48.0, 0.0);
    let trunk_length = outlet.distance(junction);
    graph
        .cell_center_km
        .iter()
        .map(|&point| {
            let cost = segment_network_cost(point, outlet, junction, 0.0)
                .min(segment_network_cost(
                    point,
                    junction,
                    left_head,
                    trunk_length,
                ))
                .min(segment_network_cost(
                    point,
                    junction,
                    right_head,
                    trunk_length,
                ));
            0.01 * cost
        })
        .collect()
}

fn role_separation_surface(graph: &EvaluationSurfaceGraphV0) -> Vec<f64> {
    let outlet = DVec3::new(0.0, -80.0, 0.0);
    let junction = DVec3::new(0.0, -20.0, 0.0);
    let left_head = DVec3::new(-115.0, 75.0, 0.0);
    let center_head = DVec3::new(0.0, 75.0, 0.0);
    let right_junction = DVec3::new(105.0, 30.0, 0.0);
    let right_left_head = DVec3::new(85.0, 75.0, 0.0);
    let right_right_head = DVec3::new(135.0, 75.0, 0.0);
    let trunk = outlet.distance(junction);
    let right = junction.distance(right_junction);
    let segments = [
        (outlet, junction, 0.0),
        (junction, left_head, trunk),
        (junction, center_head, trunk),
        (junction, right_junction, trunk),
        (right_junction, right_left_head, trunk + right),
        (right_junction, right_right_head, trunk + right),
    ];
    graph
        .cell_center_km
        .iter()
        .map(|&point| {
            0.01 * segments
                .iter()
                .map(|&(start, end, offset)| segment_network_cost(point, start, end, offset))
                .fold(f64::INFINITY, f64::min)
        })
        .collect()
}

#[test]
fn d0_plane_closes_and_is_deterministic_at_8_4_2_km() {
    for spacing in [8.0, 4.0, 2.0] {
        let portal = south_portal(41, 96.0, 0.0);
        let graph = graph_with_portals(96.0, 64.0, spacing, &[portal]);
        let elevation = plane(&graph, 40.0);
        let before: Vec<_> = elevation.iter().map(|value| value.to_bits()).collect();
        let runoff = area_supply(&graph, 0.1);
        let first =
            build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default())
                .unwrap();
        let second =
            build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default())
                .unwrap();

        assert_eq!(first, second);
        assert_eq!(
            before,
            elevation
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(first.depressions.is_empty());
        assert!(first.routing.fill_supported.iter().all(|value| !value));
        assert_eq!(first.routing.portal_ledgers.len(), 1);
        assert_eq!(first.routing.portal_ledgers[0].portal_id, 41);
        assert!(first.routing.structural_area_residual_km2.abs() < 1.0e-8);
        assert!(first.routing.supplied_runoff_residual.abs() < 1.0e-8);
        assert_eq!(first.scales.len(), 3);
        assert_eq!(
            first
                .scales
                .iter()
                .map(|scale| scale.support_threshold_km2)
                .collect::<Vec<_>>(),
            vec![1_000.0, 2_000.0, 4_000.0]
        );
        assert!(first
            .routing
            .outlet_portal_id
            .iter()
            .all(|portal| *portal == 41));
    }
}

#[test]
fn d0_exact_flat_uses_potential_and_preserves_two_portals() {
    let portals = [
        OutletPortal {
            id: OutletPortalId(7),
            side: BoundarySide::South,
            span_start_km: -48.0,
            span_end_km: 0.0,
            base_level_km: 1.0,
        },
        OutletPortal {
            id: OutletPortalId(19),
            side: BoundarySide::South,
            span_start_km: 0.0,
            span_end_km: 48.0,
            base_level_km: 1.0,
        },
    ];
    for spacing in [8.0, 4.0, 2.0] {
        let graph = graph_with_portals(96.0, 64.0, spacing, &portals);
        let elevation = vec![1.0; graph.cell_count()];
        let runoff = area_supply(&graph, 0.2);
        let result =
            build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default())
                .unwrap();

        assert!(result.depressions.is_empty());
        assert!(result.routing.flat_supported.iter().any(|value| *value));
        assert!(result
            .routing
            .physically_non_descending
            .iter()
            .all(|value| *value));
        assert_eq!(
            result
                .routing
                .portal_ledgers
                .iter()
                .map(|entry| entry.portal_id)
                .collect::<Vec<_>>(),
            vec![7, 19]
        );
        assert!(result
            .routing
            .portal_ledgers
            .iter()
            .all(|entry| entry.structural_area_km2 > 0.0 && entry.supplied_runoff > 0.0));
    }
}

#[test]
fn d0_nested_fill_reports_hierarchy_without_surface_mutation() {
    let portal = south_portal(5, 256.0, 0.0);
    for spacing in [8.0, 4.0, 2.0] {
        let graph = graph_with_portals(256.0, 192.0, spacing, std::slice::from_ref(&portal));
        let mut elevation = plane(&graph, 110.0);
        for (cell, center) in graph.cell_center_km.iter().enumerate() {
            let radius = center.length();
            if radius < 24.0 {
                elevation[cell] = 10.0;
            } else if radius < 40.0 {
                elevation[cell] = 160.0;
            } else if radius < 72.0 {
                elevation[cell] = 20.0;
            }
        }
        let before: Vec<_> = elevation.iter().map(|value| value.to_bits()).collect();
        let runoff = area_supply(&graph, 0.1);
        let result =
            build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default())
                .unwrap();

        assert_eq!(
            before,
            elevation
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(result.depressions.len() >= 2, "{:?}", result.depressions);
        assert!(result
            .depressions
            .iter()
            .any(|depression| depression.parent.is_some()));
        assert!(result.depressions.iter().all(|depression| {
            depression.affected_area_km2 > 0.0
                && depression.maximum_fill_depth_km > 0.0
                && depression.virtual_fill_volume_km3 > 0.0
        }));
        assert!(result.routing.fill_supported.iter().any(|value| *value));
        assert!(result
            .routing
            .physically_non_descending
            .iter()
            .any(|value| *value));
    }
}

#[test]
fn d0_raw_boundaries_are_one_copy_partition_seams() {
    let portals = [
        OutletPortal {
            id: OutletPortalId(1),
            side: BoundarySide::South,
            span_start_km: -48.0,
            span_end_km: 0.0,
            base_level_km: 0.0,
        },
        OutletPortal {
            id: OutletPortalId(2),
            side: BoundarySide::South,
            span_start_km: 0.0,
            span_end_km: 48.0,
            base_level_km: 0.0,
        },
    ];
    let graph = graph_with_portals(96.0, 64.0, 4.0, &portals);
    let elevation = plane(&graph, 40.0);
    let runoff = area_supply(&graph, 0.1);
    let result =
        build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default())
            .unwrap();
    let reference = &result.scales[1];
    assert!(!reference.basin_graph.raw_catchment_boundaries.is_empty());
    assert_eq!(
        reference
            .basin_graph
            .exclusive_owner
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>(),
        std::collections::BTreeSet::from([
            IncrementalCatchmentOwnerV0::Portal(1),
            IncrementalCatchmentOwnerV0::Portal(2),
        ])
    );
    let expected_boundary_count = (0..graph.edge_neighbor.len())
        .filter(|&edge| {
            let reciprocal = graph.edge_reciprocal[edge] as usize;
            if edge > reciprocal {
                return false;
            }
            let owner = graph
                .edge_offsets
                .partition_point(|&offset| offset as usize <= edge)
                - 1;
            let neighbor = graph.edge_neighbor[edge] as usize;
            reference.basin_graph.exclusive_owner[owner]
                != reference.basin_graph.exclusive_owner[neighbor]
        })
        .count();
    assert_eq!(
        reference.basin_graph.raw_catchment_boundaries.len(),
        expected_boundary_count
    );
    let mut keys = std::collections::BTreeSet::new();
    for boundary in &reference.basin_graph.raw_catchment_boundaries {
        assert_ne!(boundary.owners[0], boundary.owners[1]);
        assert!(boundary.physical_length_km > 0.0);
        let key = (
            boundary.owners,
            boundary.endpoints_km[0].x.to_bits(),
            boundary.endpoints_km[0].y.to_bits(),
            boundary.endpoints_km[1].x.to_bits(),
            boundary.endpoints_km[1].y.to_bits(),
        );
        assert!(keys.insert(key));
    }
}

#[test]
fn d0_narrow_portal_retains_confluences_and_closes_incremental_catchments() {
    let portal = OutletPortal {
        id: OutletPortalId(23),
        side: BoundarySide::South,
        span_start_km: -1.0,
        span_end_km: 1.0,
        base_level_km: 0.0,
    };
    let graph = graph_with_portals(128.0, 96.0, 4.0, &[portal]);
    let elevation = plane(&graph, 60.0);
    let runoff = area_supply(&graph, 0.3);
    let result =
        build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default())
            .unwrap();
    let reference = &result.scales[1];
    assert_eq!(reference.reach_graph.reaches.len(), 3);
    assert_eq!(
        reference
            .reach_graph
            .reaches
            .iter()
            .filter(|reach| reach.upstream_reaches.len() >= 2)
            .count(),
        1
    );
    for reach in &reference.reach_graph.reaches {
        if let Some(downstream) = reach.downstream_reach {
            assert!(reference.reach_graph.reaches[downstream as usize]
                .upstream_reaches
                .contains(&reach.id));
        }
        let maximum = reach
            .upstream_reaches
            .iter()
            .map(|&id| reference.reach_graph.reaches[id as usize].strahler_order)
            .max()
            .unwrap_or(0);
        let ties = reach
            .upstream_reaches
            .iter()
            .filter(|&&id| reference.reach_graph.reaches[id as usize].strahler_order == maximum)
            .count();
        let expected = if maximum == 0 {
            1
        } else if ties >= 2 {
            maximum + 1
        } else {
            maximum
        };
        assert_eq!(reach.strahler_order, expected);
    }

    let reach_area: f64 = reference
        .basin_graph
        .catchments
        .iter()
        .map(|catchment| catchment.exclusive_physical_area_km2)
        .sum();
    let portal_area: f64 = reference
        .basin_graph
        .exclusive_owner
        .iter()
        .enumerate()
        .filter(|(_, owner)| matches!(owner, IncrementalCatchmentOwnerV0::Portal(23)))
        .map(|(cell, _)| graph.cell_area_km2[cell])
        .sum();
    let total_area: f64 = graph.cell_area_km2.iter().sum();
    assert!((reach_area + portal_area - total_area).abs() < 1.0e-8);

    let reach_runoff: f64 = reference
        .basin_graph
        .catchments
        .iter()
        .map(|catchment| catchment.exclusive_local_runoff)
        .sum();
    let portal_runoff: f64 = reference
        .basin_graph
        .exclusive_owner
        .iter()
        .enumerate()
        .filter(|(_, owner)| matches!(owner, IncrementalCatchmentOwnerV0::Portal(23)))
        .map(|(cell, _)| runoff[cell])
        .sum();
    let total_runoff: f64 = runoff.iter().sum();
    assert!((reach_runoff + portal_runoff - total_runoff).abs() < 1.0e-8);
}

#[test]
fn d0_asymmetric_fork_reports_raw_8_4_2_topology() {
    let portal = OutletPortal {
        id: OutletPortalId(23),
        side: BoundarySide::South,
        span_start_km: -1.0,
        span_end_km: 1.0,
        base_level_km: 0.0,
    };
    let mut summaries = Vec::new();
    for spacing in [8.0, 4.0, 2.0] {
        let graph = graph_with_portals(128.0, 96.0, spacing, std::slice::from_ref(&portal));
        let elevation = y_valley_surface(&graph);
        let runoff: Vec<_> = graph
            .cell_area_km2
            .iter()
            .zip(&graph.cell_center_km)
            .map(|(&area, center)| 0.3 * (1.0 + 0.002 * center.x) * area)
            .collect();
        let result =
            build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default())
                .unwrap();
        let reference = &result.scales[1];
        let confluences = reference
            .reach_graph
            .reaches
            .iter()
            .filter(|reach| reach.upstream_reaches.len() >= 2)
            .count();
        let maximum_order = reference
            .reach_graph
            .reaches
            .iter()
            .map(|reach| reach.strahler_order)
            .max()
            .unwrap_or(0);
        summaries.push((
            spacing,
            reference.reach_graph.reaches.len(),
            confluences,
            maximum_order,
            reference.basin_graph.raw_catchment_boundaries.len(),
        ));
        assert!(result.routing.structural_area_residual_km2.abs() < 1.0e-8);
        assert!(result.routing.supplied_runoff_residual.abs() < 1.0e-8);
        assert!(!reference.reach_graph.reaches.is_empty());
    }
    eprintln!("D0 asymmetric fork raw 8/4/2 summaries: {summaries:?}");
    assert!(summaries
        .iter()
        .all(|summary| (summary.1, summary.2, summary.3) == (3, 1, 2)));
}

#[test]
fn d0_public_role_fixture_separates_supply_length_and_order() {
    let portal = OutletPortal {
        id: OutletPortalId(31),
        side: BoundarySide::South,
        span_start_km: -1.0,
        span_end_km: 1.0,
        base_level_km: 0.0,
    };
    let graph = graph_with_portals(280.0, 160.0, 4.0, &[portal]);
    let elevation = role_separation_surface(&graph);
    let runoff: Vec<_> = graph
        .cell_area_km2
        .iter()
        .zip(&graph.cell_center_km)
        .map(|(&area, center)| {
            let multiplier = if center.x.abs() < 25.0 && center.y > 0.0 {
                20.0
            } else {
                1.0
            };
            multiplier * area
        })
        .collect();
    let result =
        build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default())
            .unwrap();
    let reference = &result.scales[1];
    assert_eq!(reference.reach_graph.portal_roles.len(), 1);
    let roles = &reference.reach_graph.portal_roles[0];
    eprintln!("D0 public role fixture: {roles:?}");
    assert_eq!(roles.greatest_supply, vec![2, 1, 3]);
    assert_eq!(roles.longest_trunk, vec![4, 3]);
    assert_eq!(roles.highest_order, vec![0, 1, 3]);
}

#[test]
fn d0_rejects_missing_portal_and_invalid_inputs() {
    let graph = graph_with_portals(48.0, 40.0, 4.0, &[]);
    let elevation = plane(&graph, 30.0);
    let runoff = area_supply(&graph, 0.1);
    assert_eq!(
        build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default()),
        Err(DrainageErrorV0::MissingPortal)
    );

    let portal = south_portal(1, 48.0, 0.0);
    let graph = graph_with_portals(48.0, 40.0, 4.0, &[portal]);
    let mut spherical = graph.clone();
    spherical.domain = EvaluationDomainV0::Spherical { radius_km: 6_371.0 };
    assert_eq!(
        build_evaluation_drainage_v0(&spherical, &elevation, &runoff, DrainageConfigV0::default()),
        Err(DrainageErrorV0::UnsupportedDomain)
    );
    assert_eq!(
        build_evaluation_drainage_v0(
            &graph,
            &elevation[..elevation.len() - 1],
            &runoff,
            DrainageConfigV0::default()
        ),
        Err(DrainageErrorV0::LengthMismatch("physical_elevation_km"))
    );
    let mut unregistered = DrainageConfigV0::default();
    unregistered.support_thresholds_km2[0] = 999.0;
    assert_eq!(
        build_evaluation_drainage_v0(&graph, &elevation, &runoff, unregistered),
        Err(DrainageErrorV0::UnregisteredConfiguration)
    );
    let mut elevation = plane(&graph, 30.0);
    elevation[0] = f64::NAN;
    assert_eq!(
        build_evaluation_drainage_v0(&graph, &elevation, &runoff, DrainageConfigV0::default()),
        Err(DrainageErrorV0::NonFiniteElevation)
    );
    elevation[0] = 0.0;
    let mut invalid_runoff = runoff;
    invalid_runoff[0] = -1.0;
    assert_eq!(
        build_evaluation_drainage_v0(
            &graph,
            &elevation,
            &invalid_runoff,
            DrainageConfigV0::default()
        ),
        Err(DrainageErrorV0::InvalidRunoff)
    );
    let overflowing_runoff = vec![f64::MAX; graph.cell_count()];
    assert!(matches!(
        build_evaluation_drainage_v0(
            &graph,
            &elevation,
            &overflowing_runoff,
            DrainageConfigV0::default()
        ),
        Err(DrainageErrorV0::NonFiniteAccumulation { .. })
    ));
}
