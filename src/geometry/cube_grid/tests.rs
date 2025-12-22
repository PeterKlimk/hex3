use super::*;
use crate::geometry::gpu_voronoi::{build_kdtree, find_k_nearest as kiddo_find_k_nearest};
use crate::geometry::{fibonacci_sphere_points_with_rng, random_sphere_points_with_rng};
use glam::Vec3;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

fn mean_spacing(num_points: usize) -> f32 {
    if num_points == 0 {
        return 0.0;
    }
    (4.0 * std::f32::consts::PI / num_points as f32).sqrt()
}

fn gen_fibonacci(n: usize, seed: u64, jitter_scale: f32) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let jitter = mean_spacing(n) * jitter_scale;
    fibonacci_sphere_points_with_rng(n, jitter, &mut rng)
}

fn gen_random(n: usize, seed: u64) -> Vec<Vec3> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    random_sphere_points_with_rng(n, &mut rng)
}

fn res_for_target(n: usize, target_points_per_cell: f64) -> usize {
    ((n as f64 / (6.0 * target_points_per_cell)).sqrt() as usize).max(4)
}

fn brute_force_knn(points: &[Vec3], query_idx: usize, k: usize) -> Vec<usize> {
    if k == 0 || points.len() <= 1 {
        return Vec::new();
    }
    let k = k.min(points.len() - 1);
    let query = points[query_idx];

    let mut items: Vec<(f32, usize)> = Vec::with_capacity(points.len().saturating_sub(1));
    for (i, &p) in points.iter().enumerate() {
        if i == query_idx {
            continue;
        }
        items.push((unit_vec_dist_sq(p, query), i));
    }
    items.sort_unstable_by(|(da, ia), (db, ib)| da.total_cmp(db).then_with(|| ia.cmp(ib)));
    items.into_iter().take(k).map(|(_, i)| i).collect()
}

fn assert_knn_basic_invariants(points_len: usize, query_idx: usize, k: usize, out: &[usize]) {
    assert!(query_idx < points_len);
    let expected_len = if k == 0 || points_len <= 1 {
        0
    } else {
        k.min(points_len - 1)
    };
    assert_eq!(out.len(), expected_len);
    assert!(!out.iter().any(|&i| i == query_idx));
    let set: HashSet<_> = out.iter().copied().collect();
    assert_eq!(set.len(), out.len(), "duplicates in knn output");
}

fn assert_sorted_by_distance(points: &[Vec3], query_idx: usize, out: &[usize]) {
    if out.is_empty() {
        return;
    }
    let q = points[query_idx];
    let mut prev = -1.0f32;
    for &idx in out {
        let d = unit_vec_dist_sq(points[idx], q);
        assert!(d >= prev);
        prev = d;
    }
}

fn assert_set_eq(a: &[usize], b: &[usize]) {
    let sa: HashSet<_> = a.iter().copied().collect();
    let sb: HashSet<_> = b.iter().copied().collect();
    assert_eq!(sa, sb);
}

#[test]
fn cube_grid_edge_cases_do_not_panic() {
    let grid = CubeMapGrid::new(&[], 4);
    let mut scratch = grid.make_scratch();
    let mut out = Vec::new();
    grid.find_k_nearest_with_scratch_into(&[], Vec3::X, 0, 0, &mut scratch, &mut out);
    assert!(out.is_empty());

    let points = vec![Vec3::X];
    let grid = CubeMapGrid::new(&points, 4);
    let mut scratch = grid.make_scratch();
    grid.find_k_nearest_with_scratch_into(&points, points[0], 0, 12, &mut scratch, &mut out);
    assert!(out.is_empty());

    let points = vec![Vec3::X, Vec3::Y];
    let grid = CubeMapGrid::new(&points, 4);
    let mut scratch = grid.make_scratch();
    grid.find_k_nearest_with_scratch_into(&points, points[0], 0, 12, &mut scratch, &mut out);
    assert_eq!(out, vec![1]);
}

#[test]
fn cube_grid_point_cell_map_matches_projection() {
    let points = gen_fibonacci(50_000, 123, 0.05);
    let grid = CubeMapGrid::new(&points, res_for_target(points.len(), 24.0));

    for i in (0..points.len()).step_by(997) {
        assert_eq!(grid.point_index_to_cell(i), grid.point_to_cell(points[i]));
    }
}

#[test]
fn cube_grid_matches_bruteforce_small_exact() {
    let sizes = [32usize, 128, 2048];
    let ks = [1usize, 3, 12, 24];
    let res_targets = [8.0f64, 24.0, 50.0];

    for &n in &sizes {
        let points = gen_fibonacci(n, 12345, 0.1);
        for &target in &res_targets {
            let res = res_for_target(n, target);
            let grid = CubeMapGrid::new(&points, res);
            let mut scratch = grid.make_scratch();
            let mut out = Vec::new();
            let mut out_dot = Vec::new();

            for &k in &ks {
                let k = k.min(n.saturating_sub(1));
                if k == 0 {
                    continue;
                }

                for qi in (0..n).step_by((n / 16).max(1)) {
                    let expected = brute_force_knn(&points, qi, k);

                    grid.find_k_nearest_with_scratch_into(
                        &points,
                        points[qi],
                        qi,
                        k,
                        &mut scratch,
                        &mut out,
                    );
                    assert_knn_basic_invariants(n, qi, k, &out);
                    assert_sorted_by_distance(&points, qi, &out);
                    assert_set_eq(&out, &expected);

                    grid.find_k_nearest_with_scratch_into_dot_topk(
                        &points,
                        points[qi],
                        qi,
                        k,
                        &mut scratch,
                        &mut out_dot,
                    );
                    assert_knn_basic_invariants(n, qi, k, &out_dot);
                    assert_sorted_by_distance(&points, qi, &out_dot);
                    assert_eq!(out_dot, expected, "dot-topk should match brute-force order");
                }
            }
        }
    }
}

#[test]
fn cube_grid_resumable_matches_bruteforce() {
    let n = 20_000;
    let points = gen_fibonacci(n, 12345, 0.1);
    let grid = CubeMapGrid::new(&points, res_for_target(n, 24.0));
    let mut scratch = grid.make_scratch();
    let mut out = Vec::new();

    for qi in (0..n).step_by(251) {
        // Start at k=12, track up to 48 and resume.
        let status = grid.find_k_nearest_resumable_into(
            &points,
            points[qi],
            qi,
            12,
            48,
            &mut scratch,
            &mut out,
        );
        let expected12 = brute_force_knn(&points, qi, 12);
        assert_knn_basic_invariants(n, qi, 12, &out);
        assert_sorted_by_distance(&points, qi, &out);
        assert_set_eq(&out, &expected12);
        assert!(matches!(
            status,
            KnnStatus::CanResume | KnnStatus::Exhausted
        ));

        let _ = grid.resume_k_nearest_into(&points, points[qi], qi, 24, &mut scratch, &mut out);
        let expected24 = brute_force_knn(&points, qi, 24);
        assert_knn_basic_invariants(n, qi, 24, &out);
        assert_sorted_by_distance(&points, qi, &out);
        assert_set_eq(&out, &expected24);

        let _ = grid.resume_k_nearest_into(&points, points[qi], qi, 48, &mut scratch, &mut out);
        let expected48 = brute_force_knn(&points, qi, 48);
        assert_knn_basic_invariants(n, qi, 48, &out);
        assert_sorted_by_distance(&points, qi, &out);
        assert_set_eq(&out, &expected48);
    }
}

#[test]
fn cube_grid_resume_beyond_track_limit_falls_back_to_exact() {
    let n = 10_000;
    let points = gen_random(n, 999);
    let grid = CubeMapGrid::new(&points, res_for_target(n, 24.0));
    let mut scratch = grid.make_scratch();
    let mut out = Vec::new();

    let qi = 1234;
    let _ =
        grid.find_k_nearest_resumable_into(&points, points[qi], qi, 12, 12, &mut scratch, &mut out);
    let status = grid.resume_k_nearest_into(&points, points[qi], qi, 24, &mut scratch, &mut out);
    assert_eq!(status, KnnStatus::Exhausted);
    let expected24 = brute_force_knn(&points, qi, 24);
    assert_knn_basic_invariants(n, qi, 24, &out);
    assert_sorted_by_distance(&points, qi, &out);
    assert_set_eq(&out, &expected24);
}

#[test]
fn cube_grid_iterative_fetch_matches_bruteforce() {
    let n = 8192;
    let k = 48;
    let points = gen_fibonacci(n, 12345, 0.1);
    let grid = CubeMapGrid::new(&points, res_for_target(n, 24.0));
    let mut scratch = grid.make_iter_scratch();

    for qi in (0..n).step_by(257) {
        let mut query = grid.knn_query::<48>(&points, points[qi], qi, &mut scratch);
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut last_dist = -1.0f32;

        loop {
            match query.fetch() {
                Some(batch) => {
                    for &(_dot, idx_u32) in batch {
                        let idx = idx_u32 as usize;
                        assert!(seen.insert(idx), "duplicate neighbor in fetch output");
                        let d = unit_vec_dist_sq(points[idx], points[qi]);
                        assert!(d >= last_dist, "fetch output not globally ordered");
                        last_dist = d;
                        out.push(idx);
                    }
                }
                None => {
                    assert!(
                        query.is_exhausted(),
                        "fetch returned None before query exhaustion"
                    );
                    break;
                }
            }
        }

        let expected = brute_force_knn(&points, qi, k);
        assert_knn_basic_invariants(n, qi, k, &out);
        assert_sorted_by_distance(&points, qi, &out);
        assert_set_eq(&out, &expected);
    }
}

#[test]
fn cube_grid_cell_bounds_are_conservative() {
    // Sanity check: random points inside a cell are within the precomputed cap.
    use rand::Rng;

    let points = gen_fibonacci(10_000, 12345, 0.1);
    let grid = CubeMapGrid::new(&points, 32);
    let mut rng = rand::thread_rng();

    let num_cells = 6 * grid.res * grid.res;
    for _ in 0..200 {
        let cell = rng.gen_range(0..num_cells);
        let (face, iu, iv) = cell_to_face_ij(cell, grid.res);
        let u0 = (iu as f32) / grid.res as f32 * 2.0 - 1.0;
        let u1 = ((iu + 1) as f32) / grid.res as f32 * 2.0 - 1.0;
        let v0 = (iv as f32) / grid.res as f32 * 2.0 - 1.0;
        let v1 = ((iv + 1) as f32) / grid.res as f32 * 2.0 - 1.0;

        let center = grid.cell_centers[cell];
        let cap_angle = grid.cell_cos_radius[cell].clamp(-1.0, 1.0).acos() + 1e-3;

        for _ in 0..50 {
            let u = rng.gen_range(u0..u1);
            let v = rng.gen_range(v0..v1);
            let p = face_uv_to_3d(face, u, v);
            let ang = center.dot(Vec3A::from(p)).clamp(-1.0, 1.0).acos();
            assert!(
                ang <= cap_angle,
                "cell cap underestimates (ang={ang}, cap={cap_angle})"
            );
        }
    }
}

#[test]
#[ignore] // Run with: cargo test cube_grid_stress_vs_kiddo_100k --release -- --ignored --nocapture
fn cube_grid_stress_vs_kiddo_100k() {
    let n = 100_000;
    let k = 24;
    let points = gen_fibonacci(n, 12345, 0.1);
    let grid = CubeMapGrid::new(&points, res_for_target(n, 24.0));
    let (tree, entries) = build_kdtree(&points);

    let mut scratch = grid.make_scratch();
    let mut out = Vec::new();

    for qi in (0..n).step_by(997) {
        grid.find_k_nearest_with_scratch_into_dot_topk(
            &points,
            points[qi],
            qi,
            k,
            &mut scratch,
            &mut out,
        );
        let kiddo = kiddo_find_k_nearest(&tree, &entries, points[qi], qi, k);
        assert_knn_basic_invariants(n, qi, k, &out);
        assert_set_eq(&out, &kiddo);
    }
}

#[test]
#[ignore] // Run with: cargo test cube_grid_stress_vs_kiddo_1m --release -- --ignored --nocapture
fn cube_grid_stress_vs_kiddo_1m() {
    let n = 1_000_000;
    let k = 24;
    let points = gen_fibonacci(n, 12345, 0.0);
    let grid = CubeMapGrid::new(&points, res_for_target(n, 8.0));
    let (tree, entries) = build_kdtree(&points);

    let mut scratch = grid.make_scratch();
    let mut out = Vec::new();

    for qi in (0..n).step_by(100_003) {
        grid.find_k_nearest_with_scratch_into_dot_topk(
            &points,
            points[qi],
            qi,
            k,
            &mut scratch,
            &mut out,
        );
        let kiddo = kiddo_find_k_nearest(&tree, &entries, points[qi], qi, k);
        assert_knn_basic_invariants(n, qi, k, &out);
        assert_set_eq(&out, &kiddo);
    }
}
