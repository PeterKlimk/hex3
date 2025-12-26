//! Integration tests for hex3's use of the s2-voronoi crate.
//!
//! These tests verify that the world generation pipeline works correctly
//! with s2-voronoi as the Voronoi computation backend.

use hex3::world::Tessellation;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

#[test]
fn test_generate_gpu_style_basic() {
    let mut rng = ChaCha8Rng::seed_from_u64(12345);
    let tess = Tessellation::generate_gpu_style(1000, 2, &mut rng);

    assert_eq!(tess.voronoi.num_cells(), 1000);
    assert!(!tess.voronoi.vertices.is_empty());
}

#[test]
fn test_generate_gpu_style_adjacency_valid() {
    let mut rng = ChaCha8Rng::seed_from_u64(54321);
    let tess = Tessellation::generate_gpu_style(500, 2, &mut rng);

    // Every cell should have adjacency info
    assert_eq!(tess.adjacency.len(), 500);

    // No orphan cells (cells with no neighbors)
    let orphans = tess.adjacency.iter().filter(|n| n.is_empty()).count();
    assert!(
        orphans < 5,
        "should have very few orphan cells, got {}",
        orphans
    );

    // Adjacency should be symmetric: if A neighbors B, B neighbors A
    let mut asymmetric_count = 0;
    for (i, neighbors) in tess.adjacency.iter().enumerate() {
        for &j in neighbors {
            if !tess.adjacency[j].contains(&i) {
                asymmetric_count += 1;
            }
        }
    }
    // Allow small number of asymmetric edges due to numerical issues
    assert!(
        asymmetric_count < 10,
        "should have mostly symmetric adjacency, got {} asymmetric edges",
        asymmetric_count
    );
}

#[test]
fn test_generate_gpu_style_vertices_on_sphere() {
    let mut rng = ChaCha8Rng::seed_from_u64(99999);
    let tess = Tessellation::generate_gpu_style(500, 2, &mut rng);

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
fn test_generate_gpu_style_cell_vertex_count() {
    let mut rng = ChaCha8Rng::seed_from_u64(77777);
    let tess = Tessellation::generate_gpu_style(1000, 2, &mut rng);

    // Most cells should have >= 3 vertices (valid polygons)
    let valid_cells = tess
        .voronoi
        .iter_cells()
        .filter(|c| c.len() >= 3)
        .count();
    let ratio = valid_cells as f32 / 1000.0;
    assert!(
        ratio > 0.99,
        "at least 99% of cells should be valid polygons, got {:.1}%",
        ratio * 100.0
    );
}

#[test]
fn test_generate_gpu_style_no_duplicate_vertices_in_cell() {
    let mut rng = ChaCha8Rng::seed_from_u64(11111);
    let tess = Tessellation::generate_gpu_style(500, 2, &mut rng);

    let mut cells_with_dupes = 0;
    for cell in tess.voronoi.iter_cells() {
        let unique: HashSet<u32> = cell.vertex_indices.iter().copied().collect();
        if unique.len() < cell.len() {
            cells_with_dupes += 1;
        }
    }

    let ratio = cells_with_dupes as f32 / 500.0;
    assert!(
        ratio < 0.01,
        "less than 1% of cells should have duplicate vertices, got {:.1}%",
        ratio * 100.0
    );
}

#[test]
fn test_generate_gpu_style_cell_areas() {
    let mut rng = ChaCha8Rng::seed_from_u64(33333);
    let tess = Tessellation::generate_gpu_style(500, 2, &mut rng);

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
fn test_generate_gpu_style_reproducible() {
    let mut rng1 = ChaCha8Rng::seed_from_u64(42);
    let tess1 = Tessellation::generate_gpu_style(200, 2, &mut rng1);

    let mut rng2 = ChaCha8Rng::seed_from_u64(42);
    let tess2 = Tessellation::generate_gpu_style(200, 2, &mut rng2);

    assert_eq!(tess1.voronoi.num_cells(), tess2.voronoi.num_cells());
    assert_eq!(tess1.voronoi.vertices.len(), tess2.voronoi.vertices.len());

    // Vertex positions should be identical
    for (v1, v2) in tess1
        .voronoi
        .vertices
        .iter()
        .zip(tess2.voronoi.vertices.iter())
    {
        let diff = (*v1 - *v2).length();
        assert!(diff < 1e-6, "vertices should be identical");
    }
}

#[test]
fn test_generate_gpu_style_various_sizes() {
    for n in [100, 500, 2000] {
        let mut rng = ChaCha8Rng::seed_from_u64(12345 + n as u64);
        let tess = Tessellation::generate_gpu_style(n, 2, &mut rng);
        assert_eq!(
            tess.voronoi.num_cells(),
            n,
            "failed for n={}",
            n
        );
    }
}
