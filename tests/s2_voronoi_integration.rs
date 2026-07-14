//! Integration tests for hex3's use of the s2-voronoi crate.
//!
//! These tests verify that the world generation pipeline works correctly
//! with s2-voronoi as the Voronoi computation backend.

use hex3::world::Tessellation;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

fn assert_adjacency_matches_shared_boundary_edges(tess: &Tessellation) {
    for cell_idx in 0..tess.num_cells() {
        let cell = tess.voronoi.cell(cell_idx);
        let neighbors = tess.neighbors(cell_idx);
        assert_eq!(
            neighbors.len(),
            cell.vertex_indices.len(),
            "cell {cell_idx} must have one native neighbor per boundary edge"
        );

        for (edge_idx, &neighbor_idx) in neighbors.iter().enumerate() {
            assert!(neighbor_idx < tess.num_cells());
            assert_ne!(cell_idx, neighbor_idx);
            assert!(
                tess.neighbors(neighbor_idx).contains(&cell_idx),
                "adjacency must be reciprocal for {cell_idx} -> {neighbor_idx}"
            );

            let edge_start = cell.vertex_indices[edge_idx];
            let edge_end = cell.vertex_indices[(edge_idx + 1) % cell.vertex_indices.len()];
            let neighbor = tess.voronoi.cell(neighbor_idx);
            let has_reversed_edge = (0..neighbor.vertex_indices.len()).any(|i| {
                neighbor.vertex_indices[i] == edge_end
                    && neighbor.vertex_indices[(i + 1) % neighbor.vertex_indices.len()]
                        == edge_start
            });
            assert!(
                has_reversed_edge,
                "neighbor {neighbor_idx} does not own the reverse of cell {cell_idx} edge {edge_idx}"
            );
        }
    }
}

#[test]
fn test_generate_knn_clipping_basic() {
    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let tess = Tessellation::generate_knn_clipping(1000, 2, &mut rng);

    assert_eq!(tess.voronoi.num_cells(), 1000);
    assert!(!tess.voronoi.vertices.is_empty());

    let report = tess
        .voronoi_backend_report
        .as_ref()
        .expect("clipping tessellation must retain its backend report");
    assert_eq!(report.original_points, 1000);
    assert_eq!(report.effective_points, tess.num_cells());
    assert_eq!(report.preferred_cells, tess.num_cells());
    assert_eq!(report.preferred_vertices, tess.voronoi.vertices.len());
    assert!(report.preferred_strictly_valid);
    assert_eq!(report.post_repair_unpaired_edge_count, 0);
    assert_eq!(report.native_missing_neighbor_entries, 0);
    assert!(!report.degeneracy_perturbation_applied);
}

#[test]
fn test_generate_knn_clipping_adjacency_valid() {
    let mut rng = ChaCha8Rng::seed_from_u64(54321);
    let tess = Tessellation::generate_knn_clipping(500, 2, &mut rng);

    // Every cell should have adjacency info
    assert_eq!(tess.adjacency.len(), 500);

    // Strict physical adjacency permits no orphan cells.
    let orphans = tess.adjacency.iter().filter(|n| n.is_empty()).count();
    assert_eq!(orphans, 0, "physical mesh must have no orphan cells");

    // Adjacency should be symmetric: if A neighbors B, B neighbors A
    let mut asymmetric_count = 0;
    for (i, neighbors) in tess.adjacency.iter().enumerate() {
        for &j in neighbors {
            if !tess.adjacency[j].contains(&i) {
                asymmetric_count += 1;
            }
        }
    }
    assert_eq!(
        asymmetric_count, 0,
        "physical adjacency must be exactly symmetric"
    );
    assert_adjacency_matches_shared_boundary_edges(&tess);
}

#[test]
fn test_generate_knn_clipping_vertices_on_sphere() {
    let mut rng = ChaCha8Rng::seed_from_u64(99999);
    let tess = Tessellation::generate_knn_clipping(500, 2, &mut rng);

    // All vertices should be on the unit sphere
    for (i, v) in tess.voronoi.vertices.iter().enumerate() {
        let len = v.length();
        assert!(
            (len - 1.0).abs() < 1e-4,
            "vertex {} not on unit sphere: length = {}",
            i,
            len
        );
    }

    // All generators should be on the unit sphere
    for (i, g) in tess.voronoi.generators.iter().enumerate() {
        let len = g.length();
        assert!(
            (len - 1.0).abs() < 1e-4,
            "generator {} not on unit sphere: length = {}",
            i,
            len
        );
    }
}

#[test]
fn test_generate_knn_clipping_cell_vertex_count() {
    let mut rng = ChaCha8Rng::seed_from_u64(77777);
    let tess = Tessellation::generate_knn_clipping(1000, 2, &mut rng);

    // Every retained physical cell must be a valid polygon.
    let valid_cells = tess.voronoi.iter_cells().filter(|c| c.len() >= 3).count();
    assert_eq!(valid_cells, tess.num_cells());
}

#[test]
fn test_generate_knn_clipping_no_duplicate_vertices_in_cell() {
    let mut rng = ChaCha8Rng::seed_from_u64(11111);
    let tess = Tessellation::generate_knn_clipping(500, 2, &mut rng);

    let mut cells_with_dupes = 0;
    for cell in tess.voronoi.iter_cells() {
        let unique: HashSet<u32> = cell.vertex_indices.iter().copied().collect();
        if unique.len() < cell.len() {
            cells_with_dupes += 1;
        }
    }

    assert_eq!(cells_with_dupes, 0);
}

#[test]
fn test_generate_knn_clipping_cell_areas() {
    let mut rng = ChaCha8Rng::seed_from_u64(33333);
    let tess = Tessellation::generate_knn_clipping(500, 2, &mut rng);

    let areas = tess.cell_areas();
    assert_eq!(areas.len(), 500);

    // All areas should be positive
    for (i, &area) in areas.iter().enumerate() {
        assert!(area > 0.0, "cell {} has non-positive area: {}", i, area);
    }

    // Total area should be close to 4*pi (surface of unit sphere)
    let total_area: f32 = areas.iter().sum();
    let expected = 4.0 * std::f32::consts::PI;
    let diff = (total_area - expected).abs() / expected;
    assert!(
        diff < 0.05,
        "total area should be close to 4*pi, got {} (expected {}, diff {:.1}%)",
        total_area,
        expected,
        diff * 100.0
    );
}

#[test]
fn test_generate_knn_clipping_reproducible() {
    let mut rng1 = ChaCha8Rng::seed_from_u64(42);
    let tess1 = Tessellation::generate_knn_clipping(200, 2, &mut rng1);

    let mut rng2 = ChaCha8Rng::seed_from_u64(42);
    let tess2 = Tessellation::generate_knn_clipping(200, 2, &mut rng2);

    assert_eq!(
        bincode::serialize(&tess1).unwrap(),
        bincode::serialize(&tess2).unwrap(),
        "fixed input must produce a byte-identical physical mesh and report"
    );
}

#[test]
fn test_knn_clipping_consumes_effective_diagram_after_welding() {
    use glam::Vec3;
    use hex3::geometry::fibonacci_sphere_points_with_rng;

    let mut rng = ChaCha8Rng::seed_from_u64(20260714);
    let mut points = fibonacci_sphere_points_with_rng(64, 0.0, &mut rng);
    let base = points[0];
    let axis = if base.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let tangent = base.cross(axis).normalize();
    points.push((base + tangent * 5.0e-7).normalize());

    let original_points = points.len();
    let tess = Tessellation::from_points_knn_clipping(points);
    let report = tess
        .voronoi_backend_report
        .as_ref()
        .expect("clipping tessellation must retain its backend report");

    assert_eq!(report.original_points, original_points);
    assert_eq!(report.merged_points, 1);
    assert_eq!(report.effective_points, original_points - 1);
    assert_eq!(tess.num_cells(), report.effective_points);
    assert_eq!(tess.adjacency.len(), report.effective_points);
    assert!(report.preferred_strictly_valid);
    assert_eq!(report.post_repair_unpaired_edge_count, 0);
    assert_eq!(report.native_missing_neighbor_entries, 0);
    assert!(!report.degeneracy_perturbation_applied);
    assert_adjacency_matches_shared_boundary_edges(&tess);
}

#[test]
fn test_generate_knn_clipping_various_sizes() {
    for n in [100, 500, 2000] {
        let mut rng = ChaCha8Rng::seed_from_u64(12345 + n as u64);
        let tess = Tessellation::generate_knn_clipping(n, 2, &mut rng);
        assert_eq!(tess.voronoi.num_cells(), n, "failed for n={}", n);
    }
}
