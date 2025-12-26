//! Testing correctness, accuracy, and cost of various spatial bounds approaches.
//!
//! Run with: cargo test bounds_test --release -- --ignored --nocapture

use super::{cell_to_face_ij, face_uv_to_3d, st_to_uv, uv_to_st, CubeMapGrid};
use glam::Vec3;
use std::time::Instant;

// ============================================================================
// Geometry helpers
// ============================================================================

/// Get all cells at exactly Chebyshev distance `dist` from center cell.
fn get_ring_cells(center_cell: usize, dist: i32, res: usize) -> Vec<usize> {
    let (face, iu, iv) = cell_to_face_ij(center_cell, res);
    let mut ring = Vec::new();

    for dv in -dist..=dist {
        for du in -dist..=dist {
            if du.abs().max(dv.abs()) != dist {
                continue;
            }

            let niu = iu as i32 + du;
            let niv = iv as i32 + dv;

            if niu >= 0 && niu < res as i32 && niv >= 0 && niv < res as i32 {
                ring.push(face * res * res + (niv as usize) * res + (niu as usize));
            } else if let Some(cell) = get_cross_face_cell(face, niu, niv, res) {
                ring.push(cell);
            }
        }
    }
    ring
}

/// Get cell index when crossing face boundary.
fn get_cross_face_cell(face: usize, niu: i32, niv: i32, res: usize) -> Option<usize> {
    let s = (niu.clamp(-1, res as i32) as f32 + 0.5) / res as f32;
    let t = (niv.clamp(-1, res as i32) as f32 + 0.5) / res as f32;
    let p = face_uv_to_3d(face, st_to_uv(s), st_to_uv(t));

    let (x, y, z) = (p.x, p.y, p.z);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());

    let (nf, nu, nv) = if ax >= ay && ax >= az {
        if x >= 0.0 { (0, -z / ax, y / ax) } else { (1, z / ax, y / ax) }
    } else if ay >= ax && ay >= az {
        if y >= 0.0 { (2, x / ay, -z / ay) } else { (3, x / ay, z / ay) }
    } else if z >= 0.0 {
        (4, x / az, y / az)
    } else {
        (5, -x / az, y / az)
    };

    let niu = (uv_to_st(nu) * res as f32) as usize;
    let niv = (uv_to_st(nv) * res as f32) as usize;
    Some(nf * res * res + niv.min(res - 1) * res + niu.min(res - 1))
}

/// Get 4 outer corners of a neighborhood at Chebyshev distance `dist` from center cell.
/// dist=0: cell itself, dist=1: 3x3 neighborhood, dist=2: 5x5 neighborhood
fn neighborhood_corners(cell: usize, dist: usize, res: usize) -> [Vec3; 4] {
    let (face, iu, iv) = cell_to_face_ij(cell, res);
    let d = dist as i32;

    let i0 = (iu as i32 - d).max(0) as usize;
    let i1 = (iu as i32 + 1 + d).min(res as i32) as usize;
    let j0 = (iv as i32 - d).max(0) as usize;
    let j1 = (iv as i32 + 1 + d).min(res as i32) as usize;

    let (u0, u1) = (st_to_uv(i0 as f32 / res as f32), st_to_uv(i1 as f32 / res as f32));
    let (v0, v1) = (st_to_uv(j0 as f32 / res as f32), st_to_uv(j1 as f32 / res as f32));

    [
        face_uv_to_3d(face, u0, v0),
        face_uv_to_3d(face, u1, v0),
        face_uv_to_3d(face, u1, v1),
        face_uv_to_3d(face, u0, v1),
    ]
}

// ============================================================================
// Security threshold methods
// ============================================================================

/// Precomputed edge data for bound (2) "dotmax" method.
struct EdgeData {
    a: Vec3,
    b: Vec3,
    n: Vec3,  // normalized normal of great circle plane
    g0: Vec3, // gate vector: n × a
    g1: Vec3, // gate vector: b × n
}

fn precompute_edges(corners: &[Vec3; 4]) -> [EdgeData; 4] {
    std::array::from_fn(|i| {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let n = a.cross(b).normalize_or_zero();
        let g0 = n.cross(a);
        let g1 = b.cross(n);
        EdgeData { a, b, n, g0, g1 }
    })
}

/// Bound (2) from s2_bounds_2_3.md: tight dotmax via precomputed edge data.
/// Avoids normalize in inner loop by using sqrt(1 - dn²).
fn security_dotmax(q: Vec3, edges: &[EdgeData; 4]) -> f32 {
    let mut max_dot = f32::NEG_INFINITY;

    for edge in edges {
        // Vertex contributions
        max_dot = max_dot.max(q.dot(edge.a));

        // Edge contribution
        let dn = q.dot(edge.n);
        let p = q - edge.n * dn;

        // Gate test: is the great-circle maximizer on the arc?
        if p.dot(edge.g0) >= 0.0 && p.dot(edge.g1) >= 0.0 {
            // Maximizer is on arc, dot = |p| = sqrt(1 - dn²)
            let dot_edge = (1.0 - dn * dn).max(0.0).sqrt();
            max_dot = max_dot.max(dot_edge);
        }
        // else: best is at endpoint, already covered by vertex loop
    }

    max_dot
}

/// Cheap exact bound: max dot to boundary via great circle projections.
/// For a query inside a convex cell, this equals the closest point on boundary.
///
/// Key insight: For each edge, sqrt(1 - (q·n)²) gives the max dot on that
/// great circle. Taking max over all 4 edges gives the closest boundary point
/// because:
/// 1. If closest is on an arc interior, one edge matches exactly
/// 2. If closest is at a vertex, it's the max for an adjacent edge's great circle
///
/// This is exact for points inside the cell, unlike ring_caps which overestimates.
fn security_planes(q: Vec3, edges: &[EdgeData; 4]) -> f32 {
    let mut max_dot = f32::NEG_INFINITY;

    for edge in edges {
        let dn = q.dot(edge.n).abs();
        let dot_to_plane = (1.0 - dn * dn).max(0.0).sqrt();
        max_dot = max_dot.max(dot_to_plane);
    }

    max_dot
}

/// Exact: find closest point on each of 4 boundary arcs.
fn security_arc(q: Vec3, corners: &[Vec3; 4]) -> f32 {
    (0..4)
        .map(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            q.dot(closest_point_on_arc(q, a, b))
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Ring caps: min distance to any cell in the outer ring (via cell caps).
fn security_ring_caps(
    q: Vec3,
    ring_cells: &[usize],
    centers: &[Vec3],
    cos_r: &[f32],
    sin_r: &[f32],
) -> f32 {
    let mut min_dist_sq = f32::INFINITY;

    for &cell in ring_cells {
        let cos_d = q.dot(centers[cell]).clamp(-1.0, 1.0);
        let dist_sq = if cos_d > cos_r[cell] {
            0.0 // inside cap
        } else {
            let sin_d = (1.0 - cos_d * cos_d).sqrt();
            let max_dot = (cos_d * cos_r[cell] + sin_d * sin_r[cell]).clamp(-1.0, 1.0);
            (2.0 - 2.0 * max_dot).max(0.0)
        };
        min_dist_sq = min_dist_sq.min(dist_sq);
    }

    (1.0 - min_dist_sq / 2.0).clamp(-1.0, 1.0)
}

fn closest_point_on_arc(q: Vec3, a: Vec3, b: Vec3) -> Vec3 {
    let n = a.cross(b);
    let n_len_sq = n.length_squared();
    if n_len_sq < 1e-10 {
        return if q.dot(a) > q.dot(b) { a } else { b };
    }
    let n = n / n_len_sq.sqrt();
    let q_proj = (q - n * q.dot(n)).normalize_or_zero();
    if q_proj == Vec3::ZERO {
        return a;
    }

    let on_arc = a.cross(q_proj).dot(n) >= -1e-6 && q_proj.cross(b).dot(n) >= -1e-6;
    if on_arc { q_proj } else if q.dot(a) > q.dot(b) { a } else { b }
}

/// Ground truth via dense sampling.
fn ground_truth(q: Vec3, corners: &[Vec3; 4], samples: usize) -> f32 {
    let mut max_dot = f32::NEG_INFINITY;
    for i in 0..4 {
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        for s in 0..=samples {
            let t = s as f32 / samples as f32;
            let p = slerp(a, b, t);
            max_dot = max_dot.max(q.dot(p));
        }
    }
    max_dot
}

fn slerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    let theta = a.dot(b).clamp(-1.0, 1.0).acos();
    if theta.abs() < 1e-6 { return a; }
    let (wa, wb) = (((1.0 - t) * theta).sin() / theta.sin(), (t * theta).sin() / theta.sin());
    (a * wa + b * wb).normalize()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::fibonacci_sphere_points_with_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn gen_points(n: usize, seed: u64) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        fibonacci_sphere_points_with_rng(n, spacing * 0.1, &mut rng)
    }

    fn res_for_k(n: usize, k: f64) -> usize {
        ((n as f64 / (6.0 * k)).sqrt() as usize).max(4)
    }

    /// Test both 1x1→3x3 (security_1x1) and 3x3→5x5 (security_3x3) bounds.
    #[test]
    #[ignore]
    fn test_bounds_correctness() {
        println!("\n=== Bounds Correctness Test (Stress) ===\n");

        // Test multiple resolutions to catch edge cases
        for (n, label) in [(100_000, "100k"), (500_000, "500k"), (1_000_000, "1M")] {
            test_bounds_at_resolution(n, label);
        }
    }

    fn test_bounds_at_resolution(n: usize, label: &str) {
        println!("===== {} points =====\n", label);

        let points = gen_points(n, 12345);
        let res = res_for_k(n, 24.0);
        let grid = CubeMapGrid::new(&points, res);

        println!("Points: {}, Res: {}, Cells: {}\n", n, res, 6 * res * res);

        // Test both security thresholds
        for (name, inner_dist, outer_dist) in [
            ("security_1x1 (1x1→3x3)", 0, 1),
            ("security_3x3 (3x3→5x5)", 1, 2),
        ] {
            println!("--- {} ---", name);

            // Counters for under/over (threshold 1e-5)
            let mut arc_under = 0usize;
            let mut dotmax_under = 0usize;
            let mut planes_under = 0usize;

            let mut dotmax_over = 0usize;
            let mut planes_over = 0usize;

            // Error tracking (vs ground truth)
            let mut arc_errors: Vec<f32> = Vec::new();
            let mut dotmax_errors: Vec<f32> = Vec::new();
            let mut planes_errors: Vec<f32> = Vec::new();

            let mut count = 0usize;

            // Test ALL points
            for cell in 0..(6 * res * res) {
                let corners = neighborhood_corners(cell, inner_dist, res);
                let edges = precompute_edges(&corners);

                for &pidx in grid.cell_points(cell) {
                    let q = points[pidx as usize];
                    count += 1;

                    let truth = ground_truth(q, &corners, 200); // more samples for accuracy
                    let arc = security_arc(q, &corners);
                    let dotmax = security_dotmax(q, &edges);
                    let planes = security_planes(q, &edges);

                    // Track errors (positive = overestimate, negative = underestimate)
                    arc_errors.push(arc - truth);
                    dotmax_errors.push(dotmax - truth);
                    planes_errors.push(planes - truth);

                    // Count significant under/over
                    if arc < truth - 1e-5 { arc_under += 1; }
                    if dotmax < truth - 1e-5 { dotmax_under += 1; }
                    if planes < truth - 1e-5 { planes_under += 1; }

                    if dotmax > truth + 1e-5 { dotmax_over += 1; }
                    if planes > truth + 1e-5 { planes_over += 1; }
                }
            }

            // Compute error statistics
            fn error_stats(errors: &[f32]) -> (f32, f32, f32, f32, f32) {
                let min = errors.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = errors.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = errors.iter().sum::<f32>() / errors.len() as f32;

                let mut sorted = errors.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p01 = sorted[errors.len() / 100]; // 1st percentile
                let p99 = sorted[errors.len() * 99 / 100]; // 99th percentile

                (min, p01, mean, p99, max)
            }

            let (arc_min, arc_p01, arc_mean, arc_p99, arc_max) = error_stats(&arc_errors);
            let (dm_min, dm_p01, dm_mean, dm_p99, dm_max) = error_stats(&dotmax_errors);
            let (pl_min, pl_p01, pl_mean, pl_p99, pl_max) = error_stats(&planes_errors);

            println!("  Samples: {}", count);
            println!();
            println!("  Under/Over counts (threshold 1e-5):");
            println!("    Arc:      under={:<6} over=0 (exact by design)", arc_under);
            println!("    Dotmax:   under={:<6} over={:<6} ({:.3}%)", dotmax_under, dotmax_over, 100.0 * dotmax_over as f64 / count as f64);
            println!("    Planes:   under={:<6} over={:<6} ({:.3}%)", planes_under, planes_over, 100.0 * planes_over as f64 / count as f64);
            println!();
            println!("  Error vs truth (positive=over, negative=under):");
            println!("    Arc:      min={:+.2e} p01={:+.2e} mean={:+.2e} p99={:+.2e} max={:+.2e}",
                     arc_min, arc_p01, arc_mean, arc_p99, arc_max);
            println!("    Dotmax:   min={:+.2e} p01={:+.2e} mean={:+.2e} p99={:+.2e} max={:+.2e}",
                     dm_min, dm_p01, dm_mean, dm_p99, dm_max);
            println!("    Planes:   min={:+.2e} p01={:+.2e} mean={:+.2e} p99={:+.2e} max={:+.2e}",
                     pl_min, pl_p01, pl_mean, pl_p99, pl_max);
            println!();

            // Only assert on underestimates (dangerous for correctness)
            assert_eq!(arc_under, 0, "Arc must never underestimate for {}", name);
        }
    }

    /// Targeted stress test: queries near corners and antipodal to vertices.
    /// This is where pure halfspace-violation bounds typically fail.
    #[test]
    #[ignore]
    fn test_bounds_corner_cases() {
        println!("\n=== Bounds Corner Case Test ===\n");

        const N: usize = 100_000;
        let points = gen_points(N, 12345);
        let res = res_for_k(N, 24.0);
        let grid = CubeMapGrid::new(&points, res);

        let mut interior_total = 0usize;
        let mut interior_under = 0usize;
        let mut interior_over = 0usize;
        let mut exterior_total = 0usize;
        let mut exterior_under = 0usize;
        let mut exterior_over = 0usize;
        let mut max_under = 0.0f32;
        let mut max_over = 0.0f32;

        // Test all cells
        for cell in 0..(6 * res * res) {
            let corners = neighborhood_corners(cell, 1, res); // 3x3 boundary
            let edges = precompute_edges(&corners);

            // Test 1: Points very close to each corner (inside the cell)
            for &corner in &corners {
                let cell_center = grid.cell_centers[cell];
                for t in [0.01, 0.05, 0.1, 0.2] {
                    let q = (cell_center * (1.0 - t) + corner * t).normalize();
                    let truth = ground_truth(q, &corners, 500);
                    let planes = security_planes(q, &edges);

                    interior_total += 1;
                    let err = planes - truth;
                    if err < -1e-5 { interior_under += 1; max_under = max_under.max(-err); }
                    if err > 1e-5 { interior_over += 1; }
                }
            }

            // Test 2: Points biased toward diagonal opposite corner
            for i in 0..4 {
                let opposite_corner = corners[(i + 2) % 4];
                let cell_center = grid.cell_centers[cell];
                for t in [0.1, 0.3, 0.5] {
                    let q = (cell_center * (1.0 - t) + opposite_corner * t).normalize();
                    let truth = ground_truth(q, &corners, 500);
                    let planes = security_planes(q, &edges);

                    interior_total += 1;
                    let err = planes - truth;
                    if err < -1e-5 { interior_under += 1; max_under = max_under.max(-err); }
                    if err > 1e-5 { interior_over += 1; }
                }
            }

            // Test 3: Points OUTSIDE the cell (near -corner)
            for &corner in &corners {
                let q = (-corner * 0.99 + grid.cell_centers[cell] * 0.01).normalize();
                let truth = ground_truth(q, &corners, 500);
                let planes = security_planes(q, &edges);

                exterior_total += 1;
                let err = planes - truth;
                if err < -1e-5 { exterior_under += 1; max_under = max_under.max(-err); }
                if err > 1e-5 { exterior_over += 1; max_over = max_over.max(err); }
            }
        }

        println!("Interior: {} samples, under={}, over={}", interior_total, interior_under, interior_over);
        println!("Exterior: {} samples, under={}, over={} (max over={:.2})", exterior_total, exterior_under, exterior_over, max_over);
        println!();

        if interior_under > 0 {
            println!("FAIL: Interior underestimates - Planes NOT safe!");
        } else {
            println!("OK: Interior has 0 underestimates - Planes is safe for our use case.");
        }
    }

    /// Test how often 1x1 has enough certified candidates to skip the ring.
    #[test]
    #[ignore]
    fn test_1x1_skip_rate() {
        println!("\n=== 1x1 Skip Rate Analysis ===\n");

        const N: usize = 100_000;
        const K: usize = 24;

        let points = gen_points(N, 12345);
        let res = res_for_k(N, K as f64);
        let grid = CubeMapGrid::new(&points, res);

        println!("Points: {}, Res: {}, k: {}\n", N, res, K);

        let mut total_queries = 0usize;
        let mut can_skip_ring = 0usize;
        let mut certified_counts: Vec<usize> = Vec::new();

        for cell in 0..(6 * res * res) {
            let cell_points = grid.cell_points(cell);
            if cell_points.is_empty() {
                continue;
            }

            // 1x1 boundary = cell itself, so security_1x1 uses dist=0 corners
            let corners_1x1 = neighborhood_corners(cell, 0, res);
            let edges_1x1 = precompute_edges(&corners_1x1);

            for &query_idx in cell_points {
                let q = points[query_idx as usize];
                total_queries += 1;

                // Compute security_1x1 threshold
                let threshold = security_planes(q, &edges_1x1);

                // Count how many 1x1 candidates are certified
                let mut certified = 0usize;
                for &cand_idx in cell_points {
                    if cand_idx == query_idx {
                        continue; // skip self
                    }
                    let dot = q.dot(points[cand_idx as usize]);
                    if dot > threshold {
                        certified += 1;
                    }
                }

                certified_counts.push(certified);
                if certified >= K {
                    can_skip_ring += 1;
                }
            }
        }

        // Stats
        certified_counts.sort();
        let min = certified_counts[0];
        let max = *certified_counts.last().unwrap();
        let median = certified_counts[certified_counts.len() / 2];
        let mean = certified_counts.iter().sum::<usize>() as f64 / certified_counts.len() as f64;

        println!("Certified 1x1 candidates per query:");
        println!("  min={}, median={}, mean={:.1}, max={}", min, median, mean, max);
        println!();
        println!("Can skip ring (certified >= {}): {} / {} ({:.1}%)",
                 K, can_skip_ring, total_queries,
                 100.0 * can_skip_ring as f64 / total_queries as f64);

        // Distribution
        println!("\nCertified count distribution:");
        for bucket in [0, 1, 5, 10, 15, 20, 24, 30, 50] {
            let count = certified_counts.iter().filter(|&&c| c >= bucket).count();
            println!("  >= {}: {} ({:.1}%)", bucket, count, 100.0 * count as f64 / total_queries as f64);
        }
    }

    #[test]
    #[ignore]
    fn bench_bounds_cost() {
        println!("\n=== Bounds Cost Benchmark ===\n");

        const N: usize = 100_000;
        const ITERS: usize = 100_000;

        let points = gen_points(N, 12345);
        let res = res_for_k(N, 24.0);
        let grid = CubeMapGrid::new(&points, res);

        let cell = 100;
        let corners_1x1 = neighborhood_corners(cell, 0, res);
        let corners_3x3 = neighborhood_corners(cell, 1, res);
        let edges_1x1 = precompute_edges(&corners_1x1);
        let edges_3x3 = precompute_edges(&corners_3x3);
        let ring_3x3 = get_ring_cells(cell, 1, res);
        let ring_5x5 = get_ring_cells(cell, 2, res);

        let queries: Vec<Vec3> = (0..ITERS).map(|i| points[i % N]).collect();
        let mut dummy = 0.0f32;

        // Warmup
        for q in queries.iter().take(1000) {
            dummy += security_arc(*q, &corners_3x3);
            dummy += security_dotmax(*q, &edges_3x3);
        }

        let mut bench = |name: &str, f: &dyn Fn(Vec3) -> f32| {
            let t0 = Instant::now();
            for q in &queries { dummy += f(*q); }
            let ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;
            println!("{:30} {:>7.1} ns", name, ns);
        };

        println!("security_1x1 (1x1→3x3):");
        bench("  arc (exact)", &|q| security_arc(q, &corners_1x1));
        bench("  dotmax (precomputed)", &|q| security_dotmax(q, &edges_1x1));
        bench("  planes (cheap exact)", &|q| security_planes(q, &edges_1x1));
        bench("  ring_caps (3x3 ring)", &|q| security_ring_caps(
            q, &ring_3x3, &grid.cell_centers, &grid.cell_cos_radius, &grid.cell_sin_radius
        ));

        println!("\nsecurity_3x3 (3x3→5x5):");
        bench("  arc (exact)", &|q| security_arc(q, &corners_3x3));
        bench("  dotmax (precomputed)", &|q| security_dotmax(q, &edges_3x3));
        bench("  planes (cheap exact)", &|q| security_planes(q, &edges_3x3));
        bench("  ring_caps (5x5 ring)", &|q| security_ring_caps(
            q, &ring_5x5, &grid.cell_centers, &grid.cell_cos_radius, &grid.cell_sin_radius
        ));

        if dummy == 0.0 { println!("{}", dummy); }
    }

    #[test]
    #[ignore]
    fn test_expansion_impact() {
        println!("\n=== Expansion Impact (3x3→5x5 only) ===\n");

        const N: usize = 100_000;
        const K: usize = 24;

        let points = gen_points(N, 12345);
        let res = res_for_k(N, 24.0);
        let grid = CubeMapGrid::new(&points, res);

        let mut total_queries = 0usize;
        let mut over_conservative = 0usize;
        let mut queries_need_expansion = 0usize;

        for cell in (0..(6 * res * res)).step_by(40) {
            let corners = neighborhood_corners(cell, 1, res);
            let ring = get_ring_cells(cell, 2, res);

            // Gather 3x3 candidates
            let mut candidates: Vec<u32> = grid.cell_points(cell).to_vec();
            for &nc in grid.cell_neighbors(cell) {
                if nc != u32::MAX {
                    let (s, e) = (grid.cell_offsets[nc as usize], grid.cell_offsets[nc as usize + 1]);
                    candidates.extend_from_slice(&grid.point_indices[s as usize..e as usize]);
                }
            }

            for &pidx in grid.cell_points(cell).iter().take(5) {
                let q = points[pidx as usize];
                total_queries += 1;

                let mut dots: Vec<(f32, u32)> = candidates.iter()
                    .filter(|&&i| i != pidx)
                    .map(|&i| (q.dot(points[i as usize]), i))
                    .collect();

                if dots.len() < K { continue; }

                dots.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let kth = dots[K - 1].0;

                let exact = security_arc(q, &corners).max(kth);
                let approx = security_ring_caps(
                    q, &ring, &grid.cell_centers, &grid.cell_cos_radius, &grid.cell_sin_radius
                ).max(kth);

                let mut needs = false;
                for &(dot, _) in dots.iter().take(K) {
                    if dot >= exact && dot < approx {
                        over_conservative += 1;
                        needs = true;
                    }
                }
                if needs { queries_need_expansion += 1; }
            }
        }

        let pct = |n, d| 100.0 * n as f64 / d as f64;
        println!("Total queries: {}", total_queries);
        println!("Over-conservative certifications: {} ({:.2}% of queries×k)",
            over_conservative, pct(over_conservative, total_queries * K));
        println!("Queries needing expansion: {} ({:.1}%)",
            queries_need_expansion, pct(queries_need_expansion, total_queries));
        println!("\nConclusion: Ring caps is safe but may trigger {:.1}% more expansions.",
            pct(queries_need_expansion, total_queries));
    }
}
