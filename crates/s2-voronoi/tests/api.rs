//! Public API integration tests for s2-voronoi.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use s2_voronoi::{compute, UnitVec3, VoronoiError};

/// Generate random points uniformly distributed on the unit sphere.
fn random_sphere_points(n: usize, seed: u64) -> Vec<UnitVec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    random_sphere_points_with_rng(n, &mut rng)
}

fn random_sphere_points_with_rng<R: Rng>(n: usize, rng: &mut R) -> Vec<UnitVec3> {
    use std::f32::consts::PI;
    (0..n)
        .map(|_| {
            let z: f32 = rng.gen_range(-1.0..1.0);
            let theta: f32 = rng.gen_range(0.0..2.0 * PI);
            let r = (1.0 - z * z).sqrt();
            UnitVec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

/// Generate Fibonacci sphere points (more uniform than random).
fn fibonacci_sphere_points(n: usize, jitter: f32, seed: u64) -> Vec<UnitVec3> {
    use std::f32::consts::PI;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let golden_angle = PI * (3.0 - 5.0f32.sqrt());

    (0..n)
        .map(|i| {
            let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
            let radius = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f32;

            let mut x = radius * theta.cos();
            let mut z = radius * theta.sin();

            if jitter > 0.0 {
                x += rng.gen_range(-jitter..jitter);
                z += rng.gen_range(-jitter..jitter);
            }

            let len = (x * x + y * y + z * z).sqrt();
            UnitVec3::new(x / len, y / len, z / len)
        })
        .collect()
}

#[test]
fn test_compute_basic() {
    let points = random_sphere_points(100, 12345);
    let output = compute(&points).expect("compute should succeed");

    assert_eq!(output.diagram.num_cells(), 100);
    assert!(output.diagram.num_vertices() > 0);
}

#[test]
fn test_compute_small_set() {
    // Small point set (algorithm needs enough neighbors to work)
    let points = random_sphere_points(20, 12345);
    let output = compute(&points).expect("20 points should work");
    assert_eq!(output.diagram.num_cells(), 20);
}

#[test]
fn test_compute_insufficient_points() {
    let points = vec![
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
    ];
    let result = compute(&points);
    assert!(matches!(result, Err(VoronoiError::InsufficientPoints(3))));
}

#[test]
fn test_compute_octahedron() {
    // 6 axis-aligned points form an octahedron
    let points = vec![
        UnitVec3::new(1.0, 0.0, 0.0),
        UnitVec3::new(-1.0, 0.0, 0.0),
        UnitVec3::new(0.0, 1.0, 0.0),
        UnitVec3::new(0.0, -1.0, 0.0),
        UnitVec3::new(0.0, 0.0, 1.0),
        UnitVec3::new(0.0, 0.0, -1.0),
    ];
    let output = compute(&points).expect("octahedron should work");

    assert_eq!(output.diagram.num_cells(), 6);
    // Each cell should have 4 vertices (square face)
    for cell in output.diagram.iter_cells() {
        assert_eq!(cell.len(), 4, "octahedron cells should have 4 vertices");
    }
}

#[test]
fn test_compute_various_sizes() {
    for n in [10, 50, 100, 500] {
        let points = fibonacci_sphere_points(n, 0.1, 42);
        let output = compute(&points).expect(&format!("n={} should work", n));
        assert_eq!(output.diagram.num_cells(), n);
    }
}

#[test]
fn test_cell_iteration() {
    let points = random_sphere_points(50, 99999);
    let output = compute(&points).unwrap();

    let mut count = 0;
    for cell in output.diagram.iter_cells() {
        assert!(cell.generator_index < 50);
        count += 1;
    }
    assert_eq!(count, 50);
}

#[test]
fn test_vertex_indices_valid() {
    let points = random_sphere_points(100, 54321);
    let output = compute(&points).unwrap();

    let num_vertices = output.diagram.num_vertices();
    for cell in output.diagram.iter_cells() {
        for &idx in cell.vertex_indices {
            assert!(
                (idx as usize) < num_vertices,
                "vertex index {} out of bounds ({})",
                idx,
                num_vertices
            );
        }
    }
}

#[test]
fn test_diagnostics_available() {
    let points = random_sphere_points(100, 11111);
    let output = compute(&points).unwrap();

    // Diagnostics should be accessible (may or may not have issues)
    let _ = output.diagnostics.bad_cells.len();
    let _ = output.diagnostics.degenerate_cells.len();
    let _ = output.diagnostics.is_clean();
}

#[test]
fn test_generators_preserved() {
    let points = random_sphere_points(20, 77777);
    let output = compute(&points).unwrap();

    // Generators should match input points
    assert_eq!(output.diagram.generators.len(), points.len());
    for (i, (gen, orig)) in output
        .diagram
        .generators
        .iter()
        .zip(points.iter())
        .enumerate()
    {
        let diff = ((gen.x - orig.x).powi(2) + (gen.y - orig.y).powi(2) + (gen.z - orig.z).powi(2))
            .sqrt();
        assert!(diff < 1e-6, "generator {} differs from input: {}", i, diff);
    }
}

#[test]
fn test_input_types() {
    // Test that different input types work via UnitVec3Like trait
    // Use enough points for the algorithm to work
    let base_points = random_sphere_points(50, 88888);

    // Convert to array format
    let arr_points: Vec<[f32; 3]> = base_points.iter().map(|p| [p.x, p.y, p.z]).collect();
    let output = compute(&arr_points).expect("array input should work");
    assert_eq!(output.diagram.num_cells(), 50);

    // Convert to tuple format
    let tuple_points: Vec<(f32, f32, f32)> = base_points.iter().map(|p| (p.x, p.y, p.z)).collect();
    let output = compute(&tuple_points).expect("tuple input should work");
    assert_eq!(output.diagram.num_cells(), 50);
}

#[test]
#[cfg(feature = "qhull")]
fn test_qhull_available() {
    use s2_voronoi::compute_voronoi_qhull;
    use glam::Vec3;

    let points: Vec<Vec3> = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];
    let voronoi = compute_voronoi_qhull(&points);
    assert_eq!(voronoi.num_cells(), 6);
}
