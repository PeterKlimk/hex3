//! Experiments with warm-start and chunked KNN approaches.
//!
//! Goal: exploit spatial coherence when processing generators in order.
//! Adjacent generators share most neighbors - avoid redundant computation.

#[cfg(test)]
mod tests {
    use crate::geometry::sphere::fibonacci_sphere_points_with_rng;
    use crate::geometry::cube_grid::CubeMapGrid;
    use glam::{Vec3, Vec3A};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    fn fibonacci_sphere_points(n: usize) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        fibonacci_sphere_points_with_rng(n, 0.0, &mut rng)
    }

    fn to_vec3a(points: &[Vec3]) -> Vec<Vec3A> {
        points.iter().map(|&p| p.into()).collect()
    }

    /// Brute force k-nearest within a candidate pool.
    fn brute_force_knn_in_pool(
        points: &[Vec3],
        query_idx: usize,
        pool: &[u32],
        k: usize,
    ) -> Vec<usize> {
        let query = points[query_idx];
        let mut dists: Vec<(usize, f32)> = pool
            .iter()
            .map(|&idx| idx as usize)
            .filter(|&i| i != query_idx)
            .map(|i| (i, (query - points[i]).length_squared()))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Brute force k-nearest for correctness checking (global).
    fn brute_force_knn(points: &[Vec3], query_idx: usize, k: usize) -> Vec<usize> {
        let query = points[query_idx];
        let mut dists: Vec<(usize, f32)> = points
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != query_idx)
            .map(|(i, p)| (i, (query - *p).length_squared()))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Batched KNN: for each bin, brute-force within 3x3 neighborhood.
    ///
    /// Simple algorithm:
    /// 1. For each bin, collect all points in 3x3 neighborhood
    /// 2. For each point in center bin, brute-force find k-nearest in pool
    ///
    /// Assumption: 3x3 neighborhood contains all k-nearest for points in center bin.
    fn batched_knn_3x3(
        grid: &CubeMapGrid,
        points: &[Vec3],
        k: usize,
    ) -> Vec<Vec<usize>> {
        let mut results = vec![Vec::new(); points.len()];
        let num_cells = 6 * grid.res * grid.res;

        for cell in 0..num_cells {
            let center_points = grid.cell_points(cell);
            if center_points.is_empty() {
                continue;
            }

            // Collect all points in 3x3 neighborhood
            let neighbors = grid.cell_neighbors(cell);
            let mut pool: Vec<u32> = Vec::new();

            // Add center cell
            pool.extend_from_slice(center_points);

            // Add 8 neighbors
            for &neighbor_cell in neighbors.iter() {
                if neighbor_cell != u32::MAX && neighbor_cell as usize != cell {
                    pool.extend_from_slice(grid.cell_points(neighbor_cell as usize));
                }
            }

            // For each point in center bin, find k-nearest in pool
            for &query_idx in center_points {
                let knn = brute_force_knn_in_pool(points, query_idx as usize, &pool, k);
                results[query_idx as usize] = knn;
            }
        }

        results
    }

    /// Current baseline: independent KNN queries for each generator.
    fn baseline_knn_all(
        grid: &CubeMapGrid,
        points_a: &[Vec3A],
        k: usize,
    ) -> Vec<Vec<usize>> {
        let mut scratch = grid.make_scratch();
        let mut results = Vec::with_capacity(points_a.len());
        let mut out = Vec::with_capacity(k);

        for i in 0..points_a.len() {
            grid.find_k_nearest_with_scratch_into(
                points_a,
                points_a[i],
                i,
                k,
                &mut scratch,
                &mut out,
            );
            results.push(out.clone());
        }
        results
    }

    #[test]
    fn test_batched_3x3_correctness() {
        let n = 10_000;
        let k = 24;
        let points = fibonacci_sphere_points(n);

        println!("\n=== Batched 3x3 Correctness Test (n={}, k={}) ===", n, k);

        // Use a reasonable resolution
        let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&points, res);

        println!("Grid: {}x{} per face, {} total cells", res, res, 6 * res * res);
        let stats = grid.stats();
        println!("Points per cell: avg={:.1}, min={}, max={}",
            stats.avg_points_per_cell, stats.min_points_per_cell, stats.max_points_per_cell);

        let batched_results = batched_knn_3x3(&grid, &points, k);

        // Check correctness against brute force
        let mut errors = 0;
        let mut checked = 0;
        for i in (0..n).step_by(100) {
            let expected = brute_force_knn(&points, i, k);
            let got = &batched_results[i];

            let expected_set: std::collections::HashSet<_> = expected.iter().collect();
            let got_set: std::collections::HashSet<_> = got.iter().collect();

            if expected_set != got_set {
                errors += 1;
                if errors <= 5 {
                    let missing: Vec<_> = expected_set.difference(&got_set).collect();
                    let extra: Vec<_> = got_set.difference(&expected_set).collect();
                    println!("Mismatch at {}: missing {:?}, extra {:?}", i, missing, extra);
                }
            }
            checked += 1;
        }

        println!("Checked {} points, {} errors ({:.2}%)",
            checked, errors, 100.0 * errors as f64 / checked as f64);

        if errors > 0 {
            println!("NOTE: Errors expected if 3x3 neighborhood is insufficient for some points");
        }
    }

    #[test]
    #[ignore] // cargo test batched_3x3_benchmark --release -- --ignored --nocapture
    fn batched_3x3_benchmark() {
        let n = 100_000;
        let k = 24;
        let points = fibonacci_sphere_points(n);
        let points_a = to_vec3a(&points);

        println!("\n=== Batched 3x3 vs Baseline Benchmark (n={}, k={}) ===", n, k);

        // Use same resolution for both
        let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&points, res);

        println!("Grid: {}x{} per face", res, res);
        let stats = grid.stats();
        println!("Points per cell: avg={:.1}, min={}, max={}",
            stats.avg_points_per_cell, stats.min_points_per_cell, stats.max_points_per_cell);

        // Warm up
        let _ = baseline_knn_all(&grid, &points_a, k);
        let _ = batched_knn_3x3(&grid, &points, k);

        // Benchmark baseline
        let t0 = Instant::now();
        let baseline_results = baseline_knn_all(&grid, &points_a, k);
        let baseline_time = t0.elapsed();
        println!("\nBaseline (heap-based): {:?} ({:.2}µs/query)",
            baseline_time, baseline_time.as_secs_f64() * 1e6 / n as f64);

        // Benchmark batched
        let t0 = Instant::now();
        let batched_results = batched_knn_3x3(&grid, &points, k);
        let batched_time = t0.elapsed();
        println!("Batched 3x3: {:?} ({:.2}µs/query)",
            batched_time, batched_time.as_secs_f64() * 1e6 / n as f64);

        let speedup = baseline_time.as_secs_f64() / batched_time.as_secs_f64();
        println!("\nSpeedup: {:.2}x", speedup);

        // Check correctness (sample)
        let mut errors = 0;
        for i in (0..n).step_by(1000) {
            let baseline_set: std::collections::HashSet<_> = baseline_results[i].iter().collect();
            let batched_set: std::collections::HashSet<_> = batched_results[i].iter().collect();
            if baseline_set != batched_set {
                errors += 1;
            }
        }
        println!("Correctness: {} errors in {} samples", errors, n / 1000);
    }

    /// SoA pool for SIMD-friendly distance computation
    struct SoaPool {
        x: Vec<f32>,
        y: Vec<f32>,
        z: Vec<f32>,
        indices: Vec<u32>,
    }

    impl SoaPool {
        fn new() -> Self {
            Self {
                x: Vec::new(),
                y: Vec::new(),
                z: Vec::new(),
                indices: Vec::new(),
            }
        }

        fn clear(&mut self) {
            self.x.clear();
            self.y.clear();
            self.z.clear();
            self.indices.clear();
        }

        fn push(&mut self, idx: u32, p: Vec3) {
            self.x.push(p.x);
            self.y.push(p.y);
            self.z.push(p.z);
            self.indices.push(idx);
        }

        fn len(&self) -> usize {
            self.indices.len()
        }

        /// Compute distances from query to all pool points.
        /// Returns (index, dist_sq) pairs.
        /// Uses chord distance squared for better numerical stability.
        #[inline]
        fn compute_distances(&self, query: Vec3, query_idx: usize) -> Vec<(u32, f32)> {
            let qx = query.x;
            let qy = query.y;
            let qz = query.z;
            let n = self.len();

            let mut results = Vec::with_capacity(n);

            for i in 0..n {
                let idx = self.indices[i];
                if idx as usize == query_idx {
                    continue;
                }
                // Use chord distance squared (more stable than 2-2*dot for small distances)
                let dx = self.x[i] - qx;
                let dy = self.y[i] - qy;
                let dz = self.z[i] - qz;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                results.push((idx, dist_sq));
            }

            results
        }

        /// Find k-nearest using partial sort (selection algorithm).
        #[inline]
        fn find_k_nearest(&self, query: Vec3, query_idx: usize, k: usize) -> Vec<usize> {
            let mut dists = self.compute_distances(query, query_idx);

            if dists.len() <= k {
                dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                return dists.iter().map(|(idx, _)| *idx as usize).collect();
            }

            // Partial sort: select k smallest, then sort them
            dists.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k);
            dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            dists.iter().map(|(idx, _)| *idx as usize).collect()
        }
    }

    /// Batched KNN with SIMD-friendly SoA layout and partial sort.
    /// No fallback - assumes 3x3 neighborhood always contains k-NN (may be wrong).
    fn batched_knn_simd(
        grid: &CubeMapGrid,
        points: &[Vec3],
        k: usize,
    ) -> Vec<Vec<usize>> {
        let mut results = vec![Vec::new(); points.len()];
        let num_cells = 6 * grid.res * grid.res;

        // Reusable pool to avoid allocations
        let mut pool = SoaPool::new();

        for cell in 0..num_cells {
            let center_points = grid.cell_points(cell);
            if center_points.is_empty() {
                continue;
            }

            // Build pool from 3x3 neighborhood (with dedup for cross-face corners)
            pool.clear();

            // Add center cell
            for &idx in center_points {
                pool.push(idx, points[idx as usize]);
            }

            // Add 8 neighbors (deduplicate cross-face corners)
            let neighbors = grid.cell_neighbors(cell);
            for (i, &neighbor_cell) in neighbors.iter().enumerate() {
                if neighbor_cell != u32::MAX && neighbor_cell as usize != cell {
                    let already_added = neighbors[..i].iter()
                        .any(|&prev| prev == neighbor_cell);
                    if !already_added {
                        for &idx in grid.cell_points(neighbor_cell as usize) {
                            pool.push(idx, points[idx as usize]);
                        }
                    }
                }
            }

            // For each point in center bin, find k-nearest in pool
            for &query_idx in center_points {
                let knn = pool.find_k_nearest(points[query_idx as usize], query_idx as usize, k);
                results[query_idx as usize] = knn;
            }
        }

        results
    }

    /// Tiled KNN: load 5x5 for cache, but each query searches only its 3x3.
    ///
    /// Each tile covers a 3x3 region of bins (stride 3).
    /// Cache benefit: load 5x5 once, process 9 cells, each doing 3x3 search.
    fn batched_knn_tiled_5x5(
        grid: &CubeMapGrid,
        points: &[Vec3],
        k: usize,
    ) -> Vec<Vec<usize>> {
        let mut results = vec![Vec::new(); points.len()];
        let res = grid.res;
        let mut pool = SoaPool::new();

        // Iterate over tiles: each tile anchored at (face, s, t) with s,t multiples of 3
        for face in 0..6 {
            let mut t_anchor = 0;
            while t_anchor < res {
                let mut s_anchor = 0;
                while s_anchor < res {
                    // Collect inner 3x3 cells (the cells we'll process)
                    let mut inner_cells = Vec::new();
                    for dt in 0..3 {
                        for ds in 0..3 {
                            let s = s_anchor + ds;
                            let t = t_anchor + dt;
                            if s < res && t < res {
                                let cell = face * res * res + t * res + s;
                                inner_cells.push(cell);
                            }
                        }
                    }

                    // For each inner cell, do 3x3 search (like SIMD 3x3)
                    // But we're iterating in spatial order for cache locality
                    for &center_cell in &inner_cells {
                        let center_points = grid.cell_points(center_cell);
                        if center_points.is_empty() {
                            continue;
                        }

                        // Build pool from this cell's 3x3 neighborhood
                        pool.clear();
                        for &idx in center_points {
                            pool.push(idx, points[idx as usize]);
                        }

                        let neighbors = grid.cell_neighbors(center_cell);
                        let mut seen = [u32::MAX; 8];
                        let mut seen_count = 0;
                        for &neighbor in neighbors.iter() {
                            if neighbor != u32::MAX && neighbor as usize != center_cell {
                                // Dedup cross-face corners
                                let already = (0..seen_count).any(|i| seen[i] == neighbor);
                                if !already {
                                    seen[seen_count] = neighbor;
                                    seen_count += 1;
                                    for &idx in grid.cell_points(neighbor as usize) {
                                        pool.push(idx, points[idx as usize]);
                                    }
                                }
                            }
                        }

                        // Query each point against its 3x3 pool
                        for &query_idx in center_points {
                            let knn = pool.find_k_nearest(
                                points[query_idx as usize],
                                query_idx as usize,
                                k,
                            );
                            results[query_idx as usize] = knn;
                        }
                    }

                    s_anchor += 3;
                }
                t_anchor += 3;
            }
        }

        results
    }

    #[test]
    #[ignore] // cargo test batched_simd_benchmark --release -- --ignored --nocapture
    fn batched_simd_benchmark() {
        let n = 100_000;
        let k = 24;
        let points = fibonacci_sphere_points(n);
        let points_a = to_vec3a(&points);

        println!("\n=== Batched SIMD vs Baseline Benchmark (n={}, k={}) ===", n, k);

        let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&points, res);

        println!("Grid: {}x{} per face", res, res);
        let stats = grid.stats();
        println!("Points per cell: avg={:.1}, min={}, max={}",
            stats.avg_points_per_cell, stats.min_points_per_cell, stats.max_points_per_cell);

        // Warm up
        let _ = baseline_knn_all(&grid, &points_a, k);
        let _ = batched_knn_simd(&grid, &points, k);

        // Benchmark baseline
        let t0 = Instant::now();
        let baseline_results = baseline_knn_all(&grid, &points_a, k);
        let baseline_time = t0.elapsed();
        println!("\nBaseline (heap-based): {:?} ({:.2}µs/query)",
            baseline_time, baseline_time.as_secs_f64() * 1e6 / n as f64);

        // Benchmark naive batched (for comparison)
        let t0 = Instant::now();
        let _ = batched_knn_3x3(&grid, &points, k);
        let naive_time = t0.elapsed();
        println!("Batched naive: {:?} ({:.2}µs/query)",
            naive_time, naive_time.as_secs_f64() * 1e6 / n as f64);

        // Benchmark SIMD batched
        let t0 = Instant::now();
        let simd_results = batched_knn_simd(&grid, &points, k);
        let simd_time = t0.elapsed();
        println!("Batched SIMD: {:?} ({:.2}µs/query)",
            simd_time, simd_time.as_secs_f64() * 1e6 / n as f64);

        println!("\nSpeedup vs baseline: {:.2}x", baseline_time.as_secs_f64() / simd_time.as_secs_f64());
        println!("Speedup vs naive: {:.2}x", naive_time.as_secs_f64() / simd_time.as_secs_f64());

        // Check correctness
        let mut errors = 0;
        for i in (0..n).step_by(1000) {
            let baseline_set: std::collections::HashSet<_> = baseline_results[i].iter().collect();
            let simd_set: std::collections::HashSet<_> = simd_results[i].iter().collect();
            if baseline_set != simd_set {
                errors += 1;
            }
        }
        println!("\nCorrectness: {} errors in {} samples", errors, n / 1000);
    }

    #[test]
    fn test_pool_size_analysis() {
        // Analyze how big the 3x3 pools are
        let n = 100_000;
        let points = fibonacci_sphere_points(n);

        println!("\n=== Pool Size Analysis ===");

        for &target_density in &[8.0, 16.0, 24.0, 32.0] {
            let res = ((n as f64 / (6.0 * target_density)).sqrt() as usize).max(4);
            let grid = CubeMapGrid::new(&points, res);
            let num_cells = 6 * res * res;

            let mut pool_sizes = Vec::new();
            for cell in 0..num_cells {
                let center_points = grid.cell_points(cell);
                if center_points.is_empty() {
                    continue;
                }

                let neighbors = grid.cell_neighbors(cell);
                let mut pool_size = center_points.len();
                for &neighbor_cell in neighbors.iter() {
                    if neighbor_cell != u32::MAX && neighbor_cell as usize != cell {
                        pool_size += grid.cell_points(neighbor_cell as usize).len();
                    }
                }
                pool_sizes.push(pool_size);
            }

            let avg = pool_sizes.iter().sum::<usize>() as f64 / pool_sizes.len() as f64;
            let min = *pool_sizes.iter().min().unwrap_or(&0);
            let max = *pool_sizes.iter().max().unwrap_or(&0);

            println!("target_density={:.0}: res={}, pool_size: avg={:.0}, min={}, max={}",
                target_density, res, avg, min, max);
        }
    }

    // ============================================================
    // CORRECTNESS-GUARANTEED BATCHED KNN WITH RING EXPANSION
    // ============================================================

    /// Compute distance from a point to its cell center, as a fraction of cell radius.
    /// Returns a value in [0, 1] where 0 = center, 1 = edge.
    /// Uses actual chord distance (not ST space) to handle S2 distortion correctly.
    #[inline]
    fn distance_from_cell_center_geometric(grid: &CubeMapGrid, cell: usize, point: Vec3) -> f32 {
        let center = grid.cell_centers[cell];
        let radius = grid.cell_cos_radius[cell].acos(); // chord distance to cell edge
        let dist_to_center = (point - center).length();
        // Clamp to [0, 1] - point should be inside cell but floating point...
        (dist_to_center / radius).min(1.0)
    }

    /// Helper: convert point to face and UV coordinates
    fn point_to_face_uv(p: Vec3) -> (usize, f32, f32) {
        let (x, y, z) = (p.x, p.y, p.z);
        let (ax, ay, az) = (x.abs(), y.abs(), z.abs());

        if ax >= ay && ax >= az {
            if x >= 0.0 {
                (0, -z / ax, y / ax)
            } else {
                (1, z / ax, y / ax)
            }
        } else if ay >= ax && ay >= az {
            if y >= 0.0 {
                (2, x / ay, -z / ay)
            } else {
                (3, x / ay, z / ay)
            }
        } else {
            if z >= 0.0 {
                (4, x / az, y / az)
            } else {
                (5, -x / az, y / az)
            }
        }
    }

    /// Compute the minimum edge-to-edge distance between adjacent cells.
    /// This is a conservative lower bound: the closest a point in ring r
    /// can be to a point in the center cell is at least (r-1) * min_edge_dist.
    ///
    /// Due to S2 quadratic projection, corner cells are smaller than center cells.
    /// We compute the minimum to ensure correctness.
    fn compute_min_cell_edge_distance(res: usize) -> f32 {
        use crate::geometry::cube_grid::{face_uv_to_3d, st_to_uv};

        let mut min_edge_dist = f32::MAX;
        let step = 1.0 / res as f32;

        // Sample cells across all faces to find minimum edge length
        for face in 0..6 {
            // Check corners and edges where distortion is worst
            for iv in [0, res / 2, res - 1] {
                for iu in [0, res / 2, res - 1] {
                    let s0 = iu as f32 * step;
                    let s1 = (iu + 1) as f32 * step;
                    let t0 = iv as f32 * step;
                    let t1 = (iv + 1) as f32 * step;

                    // Get corners
                    let corners = [
                        (s0, t0), (s0, t1), (s1, t0), (s1, t1)
                    ];
                    let corner_pts: Vec<Vec3> = corners.iter().map(|&(s, t)| {
                        let u = st_to_uv(s);
                        let v = st_to_uv(t);
                        face_uv_to_3d(face, u, v)
                    }).collect();

                    // Check edge lengths (4 edges per cell)
                    let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];
                    for (i, j) in edges {
                        let dist = (corner_pts[i] - corner_pts[j]).length();
                        min_edge_dist = min_edge_dist.min(dist);
                    }
                }
            }
        }

        min_edge_dist
    }

    /// Compute the minimum edge length for each cell in the grid.
    /// Returns a Vec where index = cell_id, value = min edge of that cell.
    fn compute_per_cell_min_edge(res: usize) -> Vec<f32> {
        use crate::geometry::cube_grid::{face_uv_to_3d, st_to_uv};

        let num_cells = 6 * res * res;
        let mut min_edges = vec![0.0f32; num_cells];
        let step = 1.0 / res as f32;

        for face in 0..6 {
            for iv in 0..res {
                for iu in 0..res {
                    let cell_id = face * res * res + iv * res + iu;

                    let s0 = iu as f32 * step;
                    let s1 = (iu + 1) as f32 * step;
                    let t0 = iv as f32 * step;
                    let t1 = (iv + 1) as f32 * step;

                    // Get corners
                    let corners = [
                        (s0, t0), (s0, t1), (s1, t0), (s1, t1)
                    ];
                    let corner_pts: [Vec3; 4] = [
                        face_uv_to_3d(face, st_to_uv(corners[0].0), st_to_uv(corners[0].1)),
                        face_uv_to_3d(face, st_to_uv(corners[1].0), st_to_uv(corners[1].1)),
                        face_uv_to_3d(face, st_to_uv(corners[2].0), st_to_uv(corners[2].1)),
                        face_uv_to_3d(face, st_to_uv(corners[3].0), st_to_uv(corners[3].1)),
                    ];

                    // Minimum of 4 edges
                    let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];
                    let mut min_edge = f32::MAX;
                    for (i, j) in edges {
                        let dist = (corner_pts[i] - corner_pts[j]).length();
                        min_edge = min_edge.min(dist);
                    }
                    min_edges[cell_id] = min_edge;
                }
            }
        }

        min_edges
    }

    /// For each cell, compute the minimum edge among itself and its 8 neighbors.
    /// This is the tightest bound for the ring-2 distance from that cell.
    fn compute_local_min_edges(grid: &CubeMapGrid) -> Vec<f32> {
        let per_cell = compute_per_cell_min_edge(grid.res);
        let num_cells = 6 * grid.res * grid.res;
        let mut local_min = vec![0.0f32; num_cells];

        for cell in 0..num_cells {
            let mut min_edge = per_cell[cell];

            // Check neighbors
            let neighbors = grid.cell_neighbors(cell);
            for &neighbor in neighbors.iter() {
                if neighbor != u32::MAX {
                    min_edge = min_edge.min(per_cell[neighbor as usize]);
                }
            }

            local_min[cell] = min_edge;
        }

        local_min
    }

    /// Compute the maximum cell diagonal (center to corner distance).
    /// This is used to determine how far a point in a cell can be from the cell center.
    fn compute_max_cell_radius(res: usize) -> f32 {
        use crate::geometry::cube_grid::{face_uv_to_3d, st_to_uv};

        let mut max_radius = 0.0f32;
        let step = 1.0 / res as f32;

        for face in 0..6 {
            for iv in [0, res / 2, res - 1] {
                for iu in [0, res / 2, res - 1] {
                    let s0 = iu as f32 * step;
                    let s1 = (iu + 1) as f32 * step;
                    let t0 = iv as f32 * step;
                    let t1 = (iv + 1) as f32 * step;

                    let u_center = st_to_uv((s0 + s1) * 0.5);
                    let v_center = st_to_uv((t0 + t1) * 0.5);
                    let center = face_uv_to_3d(face, u_center, v_center);

                    for &(s, t) in &[(s0, t0), (s0, t1), (s1, t0), (s1, t1)] {
                        let u = st_to_uv(s);
                        let v = st_to_uv(t);
                        let corner = face_uv_to_3d(face, u, v);
                        let dist = (center - corner).length();
                        max_radius = max_radius.max(dist);
                    }
                }
            }
        }

        max_radius
    }

    /// Get all cells at exactly ring distance r from a center cell.
    /// Ring 0 = just the center cell
    /// Ring 1 = 8 immediate neighbors (3x3 minus center)
    /// Ring 2 = 16 cells forming the outer edge of 5x5
    /// etc.
    fn get_ring_cells(grid: &CubeMapGrid, center_cell: usize, ring: usize) -> Vec<usize> {
        if ring == 0 {
            return vec![center_cell];
        }

        // BFS to find all cells at exactly distance `ring`
        use std::collections::{HashSet, VecDeque};

        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new(); // (cell, distance)
        let mut result = Vec::new();

        queue.push_back((center_cell, 0));
        visited.insert(center_cell);

        while let Some((cell, dist)) = queue.pop_front() {
            if dist == ring {
                result.push(cell);
                continue;
            }

            if dist > ring {
                continue;
            }

            // Explore neighbors
            let neighbors = grid.cell_neighbors(cell);
            for &neighbor in neighbors.iter() {
                if neighbor != u32::MAX {
                    let n = neighbor as usize;
                    if !visited.contains(&n) {
                        visited.insert(n);
                        queue.push_back((n, dist + 1));
                    }
                }
            }
        }

        result
    }

    /// Batched KNN with provable correctness via ring expansion.
    ///
    /// Starts with 3x3 (ring 1), then checks if any point's k-th neighbor
    /// could potentially be beaten by a point in the next ring. If so,
    /// expands and recomputes.
    fn batched_knn_correct(
        grid: &CubeMapGrid,
        points: &[Vec3],
        k: usize,
    ) -> (Vec<Vec<usize>>, BatchedKnnStats) {
        let mut results = vec![Vec::new(); points.len()];
        let num_cells = 6 * grid.res * grid.res;

        // Conservative distance bounds:
        // - min_edge_dist: minimum edge length of any cell
        // - max_cell_radius: maximum distance from cell center to any corner
        //
        // A point Q in the center cell and a point P in ring r:
        // - Q can be at most max_cell_radius from center cell's center
        // - P is in a cell that is (r-1) cells away (ring 1 = adjacent, so 0 cells gap)
        // - The minimum distance is: (r-1) * min_edge_dist - 2 * max_cell_radius
        //   (subtracting because both points can be at the edges of their cells)
        //
        // For correctness, we need to keep expanding while:
        //   worst_d_k > min_possible_dist_to_ring(r+1)
        let min_edge_dist = compute_min_cell_edge_distance(grid.res);
        let max_cell_radius = compute_max_cell_radius(grid.res);

        // Track statistics
        let mut stats = BatchedKnnStats::default();

        // Reusable pool
        let mut pool = SoaPool::new();

        for cell in 0..num_cells {
            let center_points = grid.cell_points(cell);
            if center_points.is_empty() {
                continue;
            }

            stats.cells_processed += 1;
            let mut current_ring = 1; // Start with 3x3

            loop {
                // Build pool from all cells up to current_ring
                pool.clear();
                for r in 0..=current_ring {
                    let ring_cells = get_ring_cells(grid, cell, r);
                    for ring_cell in ring_cells {
                        for &idx in grid.cell_points(ring_cell) {
                            pool.push(idx, points[idx as usize]);
                        }
                    }
                }

                // Find k-nearest for all points in center cell
                let mut worst_d_k: f32 = 0.0;

                for &query_idx in center_points {
                    let knn = pool.find_k_nearest(points[query_idx as usize], query_idx as usize, k);

                    // Get the k-th neighbor distance (chord distance, not squared)
                    if knn.len() == k {
                        let d_k = (points[query_idx as usize] - points[knn[k - 1]]).length();
                        worst_d_k = worst_d_k.max(d_k);
                    } else {
                        // Not enough neighbors yet - must expand
                        worst_d_k = f32::MAX;
                    }

                    results[query_idx as usize] = knn;
                }

                // Check if we need to expand
                // Ring (current_ring + 1) cells are (current_ring + 1) hops from center.
                // They're separated from center by current_ring layers of cells.
                //
                // Minimum distance to any point in ring (current_ring + 1):
                // - The "gap" is current_ring cell edges
                // - Subtract one edge as margin for corner effects
                //
                // Ring 1 included → next is ring 2, min_dist ≈ 0 (adjacent via corner)
                // Ring 2 included → next is ring 3, min_dist ≈ 1 edge
                // Ring 3 included → next is ring 4, min_dist ≈ 2 edges
                let min_dist_to_next_ring = ((current_ring as f32 - 1.0) * min_edge_dist).max(0.0);

                if worst_d_k < min_dist_to_next_ring && worst_d_k < f32::MAX {
                    // All points are satisfied - no point in next ring can beat current k-th
                    if current_ring > 1 {
                        stats.expanded_to_5x5 += 1;
                    }
                    if current_ring > 2 {
                        stats.expanded_to_7x7_plus += 1;
                    }
                    stats.max_ring = stats.max_ring.max(current_ring);
                    break;
                }

                // Need to expand
                current_ring += 1;

                // Safety limit
                if current_ring > 20 {
                    stats.hit_max_ring += 1;
                    break;
                }
            }
        }

        (results, stats)
    }

    #[derive(Default, Debug)]
    struct BatchedKnnStats {
        cells_processed: usize,
        expanded_to_5x5: usize,
        expanded_to_7x7_plus: usize,
        hit_max_ring: usize,
        max_ring: usize,
    }

    // ============================================================
    // TESTS FOR WEIRD POINT DISTRIBUTIONS
    // ============================================================

    /// Generate points clustered in a small region (stress test for sparse bins)
    fn clustered_points(n: usize, cluster_center: Vec3, cluster_radius: f32, seed: u64) -> Vec<Vec3> {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            // Random direction
            let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
            let phi = (1.0 - 2.0 * rng.gen::<f32>()).acos();
            let r = rng.gen::<f32>().powf(1.0 / 3.0) * cluster_radius;

            let offset = Vec3::new(
                r * phi.sin() * theta.cos(),
                r * phi.sin() * theta.sin(),
                r * phi.cos(),
            );

            let point = (cluster_center + offset).normalize();
            points.push(point);
        }
        points
    }

    /// Generate points along a great circle (1D distribution on sphere)
    fn great_circle_points(n: usize, seed: u64) -> Vec<Vec3> {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Pick a random great circle (equator rotated)
        let axis = Vec3::new(
            rng.gen::<f32>() - 0.5,
            rng.gen::<f32>() - 0.5,
            rng.gen::<f32>() - 0.5,
        ).normalize();

        let mut points = Vec::with_capacity(n);
        for i in 0..n {
            // Small random offset from exact great circle
            let t = (i as f32 / n as f32) * 2.0 * std::f32::consts::PI;
            let noise = (rng.gen::<f32>() - 0.5) * 0.1;

            // Point on great circle perpendicular to axis
            let u = if axis.x.abs() < 0.9 {
                axis.cross(Vec3::X).normalize()
            } else {
                axis.cross(Vec3::Y).normalize()
            };
            let v = axis.cross(u);

            let point = (u * (t + noise).cos() + v * (t + noise).sin()).normalize();
            points.push(point);
        }
        points
    }

    /// Generate two antipodal clusters
    fn antipodal_clusters(n: usize, seed: u64) -> Vec<Vec3> {
        let half = n / 2;
        let mut points = clustered_points(half, Vec3::Z, 0.3, seed);
        points.extend(clustered_points(n - half, -Vec3::Z, 0.3, seed + 1));
        points
    }

    /// Points concentrated at cube corners (max S2 distortion)
    fn cube_corner_points(n: usize, seed: u64) -> Vec<Vec3> {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let corners = [
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(1.0, 1.0, -1.0).normalize(),
            Vec3::new(1.0, -1.0, 1.0).normalize(),
            Vec3::new(1.0, -1.0, -1.0).normalize(),
            Vec3::new(-1.0, 1.0, 1.0).normalize(),
            Vec3::new(-1.0, 1.0, -1.0).normalize(),
            Vec3::new(-1.0, -1.0, 1.0).normalize(),
            Vec3::new(-1.0, -1.0, -1.0).normalize(),
        ];

        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            let corner = corners[rng.gen_range(0..8)];
            let offset = Vec3::new(
                (rng.gen::<f32>() - 0.5) * 0.2,
                (rng.gen::<f32>() - 0.5) * 0.2,
                (rng.gen::<f32>() - 0.5) * 0.2,
            );
            points.push((corner + offset).normalize());
        }
        points
    }

    /// Very non-uniform: most points in one hemisphere
    fn hemisphere_heavy(n: usize, seed: u64) -> Vec<Vec3> {
        use rand::Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let mut points = Vec::with_capacity(n);
        for _ in 0..n {
            // 90% in northern hemisphere, 10% in southern
            let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
            let z = if rng.gen::<f32>() < 0.9 {
                rng.gen::<f32>() // 0 to 1 (northern)
            } else {
                -rng.gen::<f32>() // -1 to 0 (southern)
            };
            let r = (1.0 - z * z).sqrt();
            points.push(Vec3::new(r * theta.cos(), r * theta.sin(), z));
        }
        points
    }

    /// Hybrid batched KNN: SIMD 3x3 with fallback to heap-based for edge cases.
    ///
    /// For each cell:
    /// 1. Find k-NN using fast SIMD 3x3 pool
    /// 2. Check if k-th distance < threshold (guarantees correctness)
    /// 3. If not, fall back to heap-based KNN for those queries
    ///
    /// Uses per-cell minimum edges (accounting for S2 distortion).
    /// Conservative bound: threshold = (1 + (0.5 - dist_from_center)) * min_edge
    /// where min_edge is the minimum among all 9 cells in the 3x3 neighborhood.
    fn batched_knn_hybrid(
        grid: &CubeMapGrid,
        points: &[Vec3],
        points_a: &[Vec3A],
        k: usize,
    ) -> (Vec<Vec<usize>>, HybridStats) {
        let mut results = vec![Vec::new(); points.len()];
        let num_cells = 6 * grid.res * grid.res;

        // For each cell, compute min edge among itself and 8 neighbors
        let local_min_edges = compute_local_min_edges(grid);

        let mut stats = HybridStats::default();
        let mut pool = SoaPool::new();
        let mut scratch = grid.make_scratch();
        let mut fallback_out = Vec::with_capacity(k);

        for cell in 0..num_cells {
            let center_points = grid.cell_points(cell);
            if center_points.is_empty() {
                continue;
            }

            stats.cells_processed += 1;

            // Minimum edge among all 9 cells in 3x3 neighborhood
            let min_edge = local_min_edges[cell];

            // Build pool from 3x3 neighborhood (deduplicate neighbor cells)
            pool.clear();
            for &idx in center_points {
                pool.push(idx, points[idx as usize]);
            }
            let neighbors = grid.cell_neighbors(cell);
            for (i, &neighbor_cell) in neighbors.iter().enumerate() {
                if neighbor_cell != u32::MAX && neighbor_cell as usize != cell {
                    // Check if we've already added this cell (cross-face corners can duplicate)
                    let already_added = neighbors[..i].iter()
                        .any(|&prev| prev == neighbor_cell);
                    if !already_added {
                        for &idx in grid.cell_points(neighbor_cell as usize) {
                            pool.push(idx, points[idx as usize]);
                        }
                    }
                }
            }

            // Find k-NN for each point in center cell
            for &query_idx in center_points {
                let query = points[query_idx as usize];
                let knn = pool.find_k_nearest(query, query_idx as usize, k);

                // Check if we need fallback
                // Conservative bound: if k-th distance >= threshold, might miss ring-2 points.
                // Use very conservative coefficients since spherical geometry is complex:
                // - Point at center (dist=0): threshold = 0.75 * min_edge
                // - Point at edge (dist=1): threshold = 0.25 * min_edge
                let needs_fallback = if knn.len() < k {
                    true // Not enough neighbors in pool
                } else {
                    let d_k = (query - points[knn[k - 1]]).length();
                    let dist_ratio = distance_from_cell_center_geometric(grid, cell, query);
                    let threshold = (0.75 - 0.5 * dist_ratio) * min_edge;
                    d_k >= threshold
                };

                if needs_fallback {
                    stats.fallbacks += 1;
                    // Use heap-based KNN
                    grid.find_k_nearest_with_scratch_into(
                        points_a,
                        points_a[query_idx as usize],
                        query_idx as usize,
                        k,
                        &mut scratch,
                        &mut fallback_out,
                    );
                    results[query_idx as usize] = fallback_out.clone();
                } else {
                    results[query_idx as usize] = knn;
                }
            }
        }

        (results, stats)
    }

    #[derive(Default, Debug)]
    struct HybridStats {
        cells_processed: usize,
        fallbacks: usize,
    }

    #[test]
    #[ignore] // cargo test batched_hybrid_benchmark --release -- --ignored --nocapture
    fn batched_hybrid_benchmark() {
        let n = 100_000;
        let k = 24;
        let points = fibonacci_sphere_points(n);
        let points_a = to_vec3a(&points);

        println!("\n=== Hybrid Batched KNN Benchmark (n={}, k={}) ===", n, k);

        // Try different densities to find optimal for each method
        let densities = [12.0, 16.0, 20.0, 24.0, 32.0];

        println!("\nDensity sweep (finding optimal for each method):");
        println!("{:>8} {:>4} {:>8} {:>10} {:>10} {:>10} {:>10}",
            "density", "res", "edge", "fallback%", "simd_µs", "hybrid_µs", "base_µs");

        let mut best_simd = (f64::MAX, 0.0);
        let mut best_hybrid = (f64::MAX, 0.0);
        let mut best_baseline = (f64::MAX, 0.0);

        for &target_pts in &densities {
            let res = ((n as f64 / (6.0 * target_pts)).sqrt() as usize).max(4);
            let grid = CubeMapGrid::new(&points, res);
            let min_edge = compute_min_cell_edge_distance(res);

            // Warm up each
            let _ = batched_knn_simd(&grid, &points, k);
            let _ = batched_knn_hybrid(&grid, &points, &points_a, k);
            let _ = baseline_knn_all(&grid, &points_a, k);

            let t0 = Instant::now();
            let _ = batched_knn_simd(&grid, &points, k);
            let simd_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

            let t0 = Instant::now();
            let (_, stats) = batched_knn_hybrid(&grid, &points, &points_a, k);
            let hybrid_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

            let t0 = Instant::now();
            let _ = baseline_knn_all(&grid, &points_a, k);
            let baseline_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

            let fallback_pct = 100.0 * stats.fallbacks as f64 / n as f64;

            println!("{:>8.0} {:>4} {:>8.4} {:>10.1} {:>10.2} {:>10.2} {:>10.2}",
                target_pts, res, min_edge, fallback_pct, simd_us, hybrid_us, baseline_us);

            if simd_us < best_simd.0 {
                best_simd = (simd_us, target_pts);
            }
            if hybrid_us < best_hybrid.0 {
                best_hybrid = (hybrid_us, target_pts);
            }
            if baseline_us < best_baseline.0 {
                best_baseline = (baseline_us, target_pts);
            }
        }

        println!("\n=== Best times (each at optimal density) ===");
        println!("Baseline (heap):  {:.2}µs @ density={:.0}", best_baseline.0, best_baseline.1);
        println!("SIMD 3x3 (naive): {:.2}µs @ density={:.0}", best_simd.0, best_simd.1);
        println!("Hybrid:           {:.2}µs @ density={:.0}", best_hybrid.0, best_hybrid.1);

        println!("\n=== Speedups (vs baseline at its optimal density) ===");
        println!("SIMD 3x3 speedup:  {:.2}x", best_baseline.0 / best_simd.0);
        println!("Hybrid speedup:    {:.2}x", best_baseline.0 / best_hybrid.0);
        println!("Hybrid vs SIMD:    {:.2}x", best_simd.0 / best_hybrid.0);

        // Verify correctness at hybrid's optimal density
        let res = ((n as f64 / (6.0 * best_hybrid.1)).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&points, res);
        let baseline_results = baseline_knn_all(&grid, &points_a, k);
        let (hybrid_results, stats) = batched_knn_hybrid(&grid, &points, &points_a, k);

        println!("\n=== Correctness check (at hybrid's optimal density={:.0}) ===", best_hybrid.1);
        println!("Fallback rate: {:.2}%", 100.0 * stats.fallbacks as f64 / n as f64);

        let local_min_edges = compute_local_min_edges(&grid);

        let mut errors = 0;
        let tie_tolerance = 1e-5;
        for i in 0..n {
            let baseline_set: std::collections::HashSet<_> = baseline_results[i].iter().collect();
            let hybrid_set: std::collections::HashSet<_> = hybrid_results[i].iter().collect();
            if baseline_set != hybrid_set {
                let query = points[i];
                let d_k = if hybrid_results[i].len() == k {
                    (query - points[hybrid_results[i][k-1]]).length()
                } else { f32::MAX };
                let is_tie = baseline_set.difference(&hybrid_set).all(|&&m| {
                    let d = (query - points[m]).length();
                    (d - d_k).abs() < tie_tolerance
                });
                if !is_tie {
                    errors += 1;
                    // Debug info for first few errors
                    if errors <= 3 {
                        let cell = grid.point_to_cell(query);
                        let min_edge = local_min_edges[cell];
                        let dist_ratio = distance_from_cell_center_geometric(&grid, cell, query);
                        let threshold = (0.75 - 0.5 * dist_ratio) * min_edge;

                        let missing: Vec<_> = baseline_set.difference(&hybrid_set).map(|&&x| x).collect();
                        let extra: Vec<_> = hybrid_set.difference(&baseline_set).map(|&&x| x).collect();

                        // Check the missing points
                        let neighbors = grid.cell_neighbors(cell);

                        // Compute actual geometric distance from query to 3x3 boundary
                        let query_cell_center = grid.cell_centers[cell];
                        let dist_query_to_own_center = (query - query_cell_center).length();
                        let query_cell_radius = grid.cell_cos_radius[cell].acos();

                        // Find closest neighbor cell center
                        let mut min_dist_to_neighbor_far_edge = f32::MAX;
                        for &n in neighbors.iter() {
                            if n != u32::MAX && n as usize != cell {
                                let n_center = grid.cell_centers[n as usize];
                                let n_radius = grid.cell_cos_radius[n as usize].acos();
                                let dist_to_n_center = (query - n_center).length();
                                // Far edge of neighbor = dist to center + radius
                                let far_edge = dist_to_n_center + n_radius;
                                min_dist_to_neighbor_far_edge = min_dist_to_neighbor_far_edge.min(far_edge);
                            }
                        }

                        println!("\n  ERROR at query {}:", i);
                        println!("    cell={}, res={}", cell, res);
                        println!("    min_edge={:.6}", min_edge);
                        println!("    dist_ratio={:.4} (in [0, 1]) -- geometric", dist_ratio);
                        println!("    threshold={:.6}", threshold);
                        println!("    d_k (hybrid's k-th)={:.6}", d_k);
                        println!("    d_k < threshold? {} (should use 3x3 result)", d_k < threshold);
                        println!("    --- Actual geometry ---");
                        println!("    query to own cell center: {:.6} (cell radius={:.6})",
                            dist_query_to_own_center, query_cell_radius);
                        println!("    min dist to neighbor far edge: {:.6}", min_dist_to_neighbor_far_edge);
                        println!("    => true safe threshold should be ~{:.6}", min_dist_to_neighbor_far_edge);
                        println!("    query cell {} neighbors: {:?}", cell, neighbors);

                        for &m in &missing {
                            let d_m = (query - points[m]).length();
                            let m_cell = grid.point_to_cell(points[m]);
                            let in_3x3 = m_cell == cell || neighbors.iter().any(|&n| n != u32::MAX && n as usize == m_cell);

                            // Check actual geometric distance to the 3x3 boundary
                            // by finding the closest cell in the 3x3 to the missing point
                            let m_neighbors = grid.cell_neighbors(m_cell);
                            let missing_touches_3x3 = m_neighbors.iter().any(|&n| {
                                n != u32::MAX && (n as usize == cell || neighbors.contains(&n))
                            });

                            println!("    MISSING: {} at dist={:.6}, cell={}, in_3x3={}, touches_3x3={}",
                                m, d_m, m_cell, in_3x3, missing_touches_3x3);
                            println!("      missing cell {} neighbors: {:?}", m_cell, m_neighbors);

                            // Compute actual distance from query to missing point's cell center
                            // to understand the geometry
                            let m_cell_center = grid.cell_centers[m_cell];
                            let dist_to_m_cell_center = (query - m_cell_center).length();
                            let m_cell_radius = grid.cell_cos_radius[m_cell].acos();
                            println!("      dist to missing cell center={:.6}, cell radius={:.6}",
                                dist_to_m_cell_center, m_cell_radius);
                        }
                        for &e in &extra {
                            let d_e = (query - points[e]).length();
                            println!("    EXTRA: {} at dist={:.6}", e, d_e);
                        }

                        // Show the k-th neighbors from both
                        let baseline_k = baseline_results[i][k-1];
                        let hybrid_k = hybrid_results[i][k-1];
                        let d_baseline_k = (query - points[baseline_k]).length();
                        let d_hybrid_k = (query - points[hybrid_k]).length();
                        println!("    baseline[k-1]={} at dist={:.6}", baseline_k, d_baseline_k);
                        println!("    hybrid[k-1]={} at dist={:.6}", hybrid_k, d_hybrid_k);
                    }
                }
            }
        }
        println!("\nHard errors: {} / {} (tie-tolerant)", errors, n);
    }

    #[test]
    #[ignore] // cargo test hybrid_weird_distributions --release -- --ignored --nocapture
    fn hybrid_weird_distributions() {
        let k = 24;

        println!("\n=== Hybrid on Weird Distributions (k={}) ===", k);

        let distributions: Vec<(&str, Vec<Vec3>)> = vec![
            ("Fibonacci (uniform)", fibonacci_sphere_points(10_000)),
            ("Clustered (single)", clustered_points(10_000, Vec3::Z, 0.5, 42)),
            ("Great circle", great_circle_points(10_000, 42)),
            ("Antipodal clusters", antipodal_clusters(10_000, 42)),
            ("Cube corners", cube_corner_points(10_000, 42)),
            ("Hemisphere heavy", hemisphere_heavy(10_000, 42)),
        ];

        for (name, points) in distributions {
            let n = points.len();
            let points_a = to_vec3a(&points);
            let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
            let grid = CubeMapGrid::new(&points, res);

            let (hybrid_results, stats) = batched_knn_hybrid(&grid, &points, &points_a, k);

            // Verify against brute force (with tie tolerance)
            let mut errors = 0;
            let mut first_error_printed = false;
            let tie_tolerance = 1e-5;
            for i in (0..n).step_by(20) {
                let expected = brute_force_knn(&points, i, k);
                let got = &hybrid_results[i];
                let query = points[i];

                let expected_set: std::collections::HashSet<_> = expected.iter().collect();
                let got_set: std::collections::HashSet<_> = got.iter().collect();

                // Check if difference is just tie-breaking
                let is_tie_difference = if expected_set != got_set && got.len() == k {
                    let d_k = (query - points[got[k-1]]).length();
                    // All missing points should be at the same distance as d_k (ties)
                    expected_set.difference(&got_set).all(|&&m| {
                        let d = (query - points[m]).length();
                        (d - d_k).abs() < tie_tolerance
                    })
                } else {
                    false
                };

                if expected_set != got_set && !is_tie_difference && !first_error_printed {
                    first_error_printed = true;
                    // Debug: analyze this error
                    let query = points[i];
                    let local_min_edges = compute_local_min_edges(&grid);

                    // Find which cell the query is in using grid's method
                    let cell = grid.point_to_cell(query);

                    let min_edge = local_min_edges[cell];
                    let dist_ratio = distance_from_cell_center_geometric(&grid, cell, query);
                    let threshold = (0.75 - 0.5 * dist_ratio) * min_edge;

                    let d_k_got = (query - points[got[k-1]]).length();
                    let d_k_expected = (query - points[expected[k-1]]).length();

                    // Find missing neighbor
                    let missing: Vec<_> = expected_set.difference(&got_set).map(|&&x| x).collect();
                    let closest_missing = missing.iter()
                        .map(|&m| (m, (query - points[m]).length()))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                    println!("\n  DEBUG first error at i={}:", i);
                    println!("    query cell={} (res={})", cell, res);
                    println!("    min_edge={:.6}, dist_ratio={:.4}", min_edge, dist_ratio);
                    println!("    threshold={:.6}", threshold);
                    println!("    d_k (got)={:.6}, d_k (expected)={:.6}", d_k_got, d_k_expected);
                    println!("    got[23]={}, expected[23]={}", got[k-1], expected[k-1]);
                    // Check if got[23] is in the 3x3 pool
                    let got_k_cell = grid.point_to_cell(points[got[k-1]]);
                    let neighbors = grid.cell_neighbors(cell);
                    let got_in_3x3 = got_k_cell == cell || neighbors.iter().any(|&n| n != u32::MAX && n as usize == got_k_cell);
                    println!("    got[23]={} in cell {}, in 3x3: {}", got[k-1], got_k_cell, got_in_3x3);
                    // Compare distance calculations
                    let p_got = points[got[k-1]];
                    let chord_dist_got = (query - p_got).length();
                    let dot_got = query.dot(p_got);
                    let dist_sq_got = 2.0 - 2.0 * dot_got;
                    println!("    got[23] distances: chord={:.9}, dist_sq={:.9}", chord_dist_got, dist_sq_got);
                    if let Some((m, d)) = closest_missing {
                        // Check which cell the missing point is in using grid's method
                        let mcell = grid.point_to_cell(points[m]);
                        // Compare distance calculations for missing point
                        let p_miss = points[m];
                        let chord_dist_miss = (query - p_miss).length();
                        let dot_miss = query.dot(p_miss);
                        let dist_sq_miss = 2.0 - 2.0 * dot_miss;
                        println!("    missing[23] distances: chord={:.9}, dist_sq={:.9}", chord_dist_miss, dist_sq_miss);
                        println!("    dist_sq comparison: got={:.12} vs miss={:.12}, diff={:.12}",
                            dist_sq_got, dist_sq_miss, dist_sq_got - dist_sq_miss);
                        println!("    missing neighbor {} at dist={:.6}, in cell {} (query cell={})",
                            m, d, mcell, cell);

                        // Is the missing point's cell in the 3x3 neighborhood?
                        let neighbors = grid.cell_neighbors(cell);
                        let in_3x3 = mcell == cell || neighbors.iter().any(|&n| n != u32::MAX && n as usize == mcell);
                        println!("    missing cell in 3x3 neighborhood: {}", in_3x3);

                        // Check if missing point is actually in the grid's cell_points for its cell
                        let missing_cell_points = grid.cell_points(mcell);
                        let in_cell_points = missing_cell_points.iter().any(|&idx| idx as usize == m);
                        println!("    missing point {} in grid.cell_points({}): {}", m, mcell, in_cell_points);

                        // List all points that WOULD be in the pool (with deduplication)
                        let mut pool_indices: Vec<usize> = Vec::new();
                        for &idx in grid.cell_points(cell) {
                            pool_indices.push(idx as usize);
                        }
                        for (ni, &neighbor_cell) in neighbors.iter().enumerate() {
                            if neighbor_cell != u32::MAX && neighbor_cell as usize != cell {
                                // Skip duplicates
                                let already_added = neighbors[..ni].iter()
                                    .any(|&prev| prev == neighbor_cell);
                                if !already_added {
                                    for &idx in grid.cell_points(neighbor_cell as usize) {
                                        pool_indices.push(idx as usize);
                                    }
                                }
                            }
                        }
                        let in_pool = pool_indices.contains(&m);
                        println!("    missing point {} in constructed pool: {} (pool size={})", m, in_pool, pool_indices.len());

                        // Check for duplicates in the pool
                        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
                        let mut duplicates: Vec<usize> = Vec::new();
                        for &idx in &pool_indices {
                            if !seen.insert(idx) {
                                duplicates.push(idx);
                            }
                        }
                        if !duplicates.is_empty() {
                            println!("    DUPLICATES in pool: {:?}", duplicates);
                            // Find which cells contain the duplicate
                            for &dup in &duplicates {
                                let dup_cell = grid.point_to_cell(points[dup]);
                                println!("      point {} is in cell {}", dup, dup_cell);
                                // Check which neighbor entries add it
                                let neighbors = grid.cell_neighbors(cell);
                                println!("      neighbors of cell {}: {:?}", cell, neighbors);
                            }
                        }

                        // Compare the actual results
                        let mut pool_dists: Vec<(usize, f32)> = pool_indices.iter()
                            .filter(|&&idx| idx != i)
                            .map(|&idx| (idx, (query - points[idx]).length()))
                            .collect();
                        pool_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                        println!("    Pool top-{} vs brute force top-{}:", k, k);
                        for rank in 0..k {
                            let (pool_idx, pool_d) = pool_dists[rank];
                            let (bf_idx, bf_d) = (expected[rank], (query - points[expected[rank]]).length());
                            let match_str = if pool_idx == bf_idx { "==" } else { "!=" };
                            println!("      rank {}: pool=({}, {:.6}) {} bf=({}, {:.6})",
                                rank, pool_idx, pool_d, match_str, bf_idx, bf_d);
                        }
                    }
                }

                if expected_set != got_set && !is_tie_difference {
                    errors += 1;
                }
            }

            let status = if errors == 0 { "✓" } else { "✗" };
            println!("{} {}: {} errors, fallback_rate={:.1}%",
                status, name, errors, 100.0 * stats.fallbacks as f64 / n as f64);
        }
    }

    #[test]
    #[ignore] // cargo test batched_correct_benchmark --release -- --ignored --nocapture
    fn batched_correct_benchmark() {
        let n = 100_000;
        let k = 24;
        let points = fibonacci_sphere_points(n);
        let points_a = to_vec3a(&points);

        println!("\n=== Batched Correct vs Baseline Benchmark (n={}, k={}) ===", n, k);

        let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&points, res);

        println!("Grid: {}x{} per face", res, res);
        let min_edge = compute_min_cell_edge_distance(res);
        let max_radius = compute_max_cell_radius(res);
        println!("Min cell edge: {:.4}, max cell radius: {:.4}", min_edge, max_radius);

        // Benchmark baseline
        let t0 = Instant::now();
        let baseline_results = baseline_knn_all(&grid, &points_a, k);
        let baseline_time = t0.elapsed();
        println!("\nBaseline (heap-based): {:?} ({:.2}µs/query)",
            baseline_time, baseline_time.as_secs_f64() * 1e6 / n as f64);

        // Benchmark correct batched
        let t0 = Instant::now();
        let (correct_results, stats) = batched_knn_correct(&grid, &points, k);
        let correct_time = t0.elapsed();
        println!("Batched correct: {:?} ({:.2}µs/query)",
            correct_time, correct_time.as_secs_f64() * 1e6 / n as f64);

        println!("\nStats: {:?}", stats);
        println!("Speedup vs baseline: {:.2}x", baseline_time.as_secs_f64() / correct_time.as_secs_f64());

        // Verify correctness
        let mut errors = 0;
        for i in 0..n {
            let baseline_set: std::collections::HashSet<_> = baseline_results[i].iter().collect();
            let correct_set: std::collections::HashSet<_> = correct_results[i].iter().collect();
            if baseline_set != correct_set {
                errors += 1;
                if errors <= 5 {
                    println!("Error at {}: baseline {:?} vs correct {:?}", i, baseline_results[i], correct_results[i]);
                }
            }
        }
        println!("\nCorrectness: {} errors in {} points", errors, n);
    }

    #[test]
    #[ignore] // cargo test weird_distributions_correctness --release -- --ignored --nocapture
    fn weird_distributions_correctness() {
        let k = 24;

        println!("\n=== Weird Distributions Correctness Test (k={}) ===", k);

        let distributions: Vec<(&str, Vec<Vec3>)> = vec![
            ("Fibonacci (uniform)", fibonacci_sphere_points(10_000)),
            ("Clustered (single)", clustered_points(10_000, Vec3::Z, 0.5, 42)),
            ("Great circle", great_circle_points(10_000, 42)),
            ("Antipodal clusters", antipodal_clusters(10_000, 42)),
            ("Cube corners", cube_corner_points(10_000, 42)),
            ("Hemisphere heavy", hemisphere_heavy(10_000, 42)),
            ("Tiny cluster", clustered_points(1_000, Vec3::X, 0.1, 42)),
        ];

        for (name, points) in distributions {
            let n = points.len();
            let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
            let grid = CubeMapGrid::new(&points, res);

            let (correct_results, stats) = batched_knn_correct(&grid, &points, k);

            // Verify against brute force
            // Use distance-tolerant comparison since ties can break differently
            let mut hard_errors = 0;
            let mut tie_differences = 0;
            let check_count = n.min(500);
            let tie_tolerance = 1e-5; // Allow ties within this distance

            for i in (0..n).step_by(n / check_count.max(1)) {
                let expected = brute_force_knn(&points, i, k);
                let got = &correct_results[i];

                let expected_set: std::collections::HashSet<_> = expected.iter().collect();
                let got_set: std::collections::HashSet<_> = got.iter().collect();

                if expected_set != got_set {
                    // Check if this is a tie-breaking difference
                    let k_th_dist = if got.len() == k {
                        (points[i] - points[got[k - 1]]).length()
                    } else {
                        f32::MAX
                    };

                    let missing: Vec<_> = expected_set.difference(&got_set).map(|&&x| x).collect();
                    let is_tie = missing.iter().all(|&m| {
                        let d = (points[i] - points[m]).length();
                        (d - k_th_dist).abs() < tie_tolerance
                    });

                    if is_tie {
                        tie_differences += 1;
                    } else {
                        hard_errors += 1;
                        if hard_errors <= 3 {
                            println!("  HARD ERROR at {}: missing {:?}", i, missing);
                            for m in &missing {
                                let d = (points[i] - points[*m]).length();
                                println!("    missing {} at dist {:.6}, k-th at {:.6}, gap={:.6}",
                                    m, d, k_th_dist, k_th_dist - d);
                            }
                        }
                    }
                }
            }

            let status = if hard_errors == 0 { "✓" } else { "✗" };
            println!("{} {}: {} hard errors, {} ties, max_ring={}, expanded_5x5={}",
                status, name, hard_errors, tie_differences, stats.max_ring, stats.expanded_to_5x5);
        }
    }

    #[test]
    #[ignore] // cargo test simd_correct_comparison --release -- --ignored --nocapture
    fn simd_correct_comparison() {
        // Compare the SIMD version (fast but not guaranteed) with correct version
        let n = 50_000;
        let k = 24;

        println!("\n=== SIMD vs Correct Comparison ===");

        let distributions: Vec<(&str, Vec<Vec3>)> = vec![
            ("Fibonacci", fibonacci_sphere_points(n)),
            ("Clustered", clustered_points(n, Vec3::Z, 0.5, 42)),
            ("Great circle", great_circle_points(n, 42)),
            ("Cube corners", cube_corner_points(n, 42)),
        ];

        for (name, points) in distributions {
            let res = ((n as f64 / (6.0 * 16.0)).sqrt() as usize).max(4);
            let grid = CubeMapGrid::new(&points, res);

            let t0 = Instant::now();
            let simd_results = batched_knn_simd(&grid, &points, k);
            let simd_time = t0.elapsed();

            let t0 = Instant::now();
            let (correct_results, stats) = batched_knn_correct(&grid, &points, k);
            let correct_time = t0.elapsed();

            // Count mismatches
            let mut mismatches = 0;
            for i in 0..n {
                let simd_set: std::collections::HashSet<_> = simd_results[i].iter().collect();
                let correct_set: std::collections::HashSet<_> = correct_results[i].iter().collect();
                if simd_set != correct_set {
                    mismatches += 1;
                }
            }

            println!("{}: SIMD={:?}, Correct={:?}, mismatches={}, max_ring={}",
                name, simd_time, correct_time, mismatches, stats.max_ring);
        }
    }

    // ============================================================
    // PAIR-BASED KNN: SYMMETRIC DOT PRODUCT APPROACH
    // ============================================================

    /// Per-point k-NN tracker using fixed-size array with O(K) insert.
    /// Tracks top-k neighbors by dot product (higher = closer for unit vectors).
    struct PointTopK<const K: usize> {
        /// (neighbor_idx, dot) pairs - higher dot = closer
        entries: [(u32, f32); K],
        count: u8,
        /// Minimum dot in current top-k, for fast rejection
        worst_dot: f32,
    }

    impl<const K: usize> PointTopK<K> {
        fn new() -> Self {
            Self {
                entries: [(u32::MAX, f32::NEG_INFINITY); K],
                count: 0,
                worst_dot: f32::NEG_INFINITY,
            }
        }

        /// Try to insert a neighbor. Returns true if inserted.
        #[inline]
        fn try_insert(&mut self, neighbor_idx: u32, dot: f32) -> bool {
            // Fast reject: if full and dot <= worst, skip
            if self.count >= K as u8 && dot <= self.worst_dot {
                return false;
            }

            let count = self.count as usize;

            if count < K {
                // Not full yet - just append
                self.entries[count] = (neighbor_idx, dot);
                self.count += 1;

                // Update worst_dot if we just filled up
                if self.count as usize == K {
                    self.worst_dot = self.entries.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min);
                }
                return true;
            }

            // Full - find and replace the worst entry
            let mut worst_idx = 0;
            let mut worst_val = self.entries[0].1;
            for i in 1..K {
                if self.entries[i].1 < worst_val {
                    worst_val = self.entries[i].1;
                    worst_idx = i;
                }
            }

            self.entries[worst_idx] = (neighbor_idx, dot);

            // Recompute worst_dot
            self.worst_dot = self.entries.iter().map(|(_, d)| *d).fold(f32::INFINITY, f32::min);
            true
        }

        /// Extract sorted neighbor indices (by distance, ascending = by dot, descending).
        fn to_sorted_indices(&self) -> Vec<usize> {
            let count = self.count as usize;
            let mut entries: Vec<(u32, f32)> = self.entries[..count].to_vec();
            // Sort by dot descending (higher dot = closer)
            entries.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            entries.iter().map(|(idx, _)| *idx as usize).collect()
        }
    }

    /// Per-point k-NN tracker with LAZY worst_dot updates.
    /// Only recomputes worst_dot when a rejection might be wrong.
    struct PointTopKLazy<const K: usize> {
        entries: [(u32, f32); K],
        count: u8,
        worst_dot: f32,
        worst_valid: bool,
    }

    impl<const K: usize> PointTopKLazy<K> {
        fn new() -> Self {
            Self {
                entries: [(u32::MAX, f32::NEG_INFINITY); K],
                count: 0,
                worst_dot: f32::NEG_INFINITY,
                worst_valid: true,
            }
        }

        #[inline]
        fn recompute_worst(&mut self) {
            self.worst_dot = self.entries[..self.count as usize]
                .iter()
                .map(|(_, d)| *d)
                .fold(f32::INFINITY, f32::min);
            self.worst_valid = true;
        }

        #[inline]
        fn try_insert(&mut self, neighbor_idx: u32, dot: f32) -> bool {
            let count = self.count as usize;

            if count < K {
                // Not full - just append
                self.entries[count] = (neighbor_idx, dot);
                self.count += 1;
                // Don't update worst_dot yet - lazy
                if self.count as usize == K {
                    self.worst_valid = false;
                }
                return true;
            }

            // Full - try fast reject with possibly stale worst_dot
            if self.worst_valid && dot <= self.worst_dot {
                return false;
            }

            // If worst_dot is stale, recompute and check again
            if !self.worst_valid {
                self.recompute_worst();
                if dot <= self.worst_dot {
                    return false;
                }
            }

            // Find and replace the worst entry
            let mut worst_idx = 0;
            let mut worst_val = self.entries[0].1;
            for i in 1..K {
                if self.entries[i].1 < worst_val {
                    worst_val = self.entries[i].1;
                    worst_idx = i;
                }
            }

            self.entries[worst_idx] = (neighbor_idx, dot);
            self.worst_valid = false; // Invalidate instead of recomputing
            true
        }

        fn to_sorted_indices(&self) -> Vec<usize> {
            let count = self.count as usize;
            let mut entries: Vec<(u32, f32)> = self.entries[..count].to_vec();
            entries.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            entries.iter().map(|(idx, _)| *idx as usize).collect()
        }
    }

    /// Per-point k-NN tracker using min-heap for O(log K) insert.
    /// Min-heap keeps smallest dot at root for fast rejection check.
    struct PointTopKHeap<const K: usize> {
        /// Min-heap: smallest dot at root (index 0)
        entries: [(u32, f32); K],
        count: u8,
    }

    impl<const K: usize> PointTopKHeap<K> {
        fn new() -> Self {
            Self {
                entries: [(u32::MAX, f32::NEG_INFINITY); K],
                count: 0,
            }
        }

        /// Peek at worst (smallest) dot - O(1)
        #[inline]
        fn worst_dot(&self) -> f32 {
            if self.count == 0 {
                f32::NEG_INFINITY
            } else {
                self.entries[0].1 // Root of min-heap
            }
        }

        /// Try to insert a neighbor - O(log K)
        #[inline]
        fn try_insert(&mut self, neighbor_idx: u32, dot: f32) -> bool {
            let count = self.count as usize;

            if count < K {
                // Not full - add to end and bubble up
                self.entries[count] = (neighbor_idx, dot);
                self.count += 1;
                self.bubble_up(count);
                return true;
            }

            // Full - check if better than worst (root)
            if dot <= self.entries[0].1 {
                return false;
            }

            // Replace root with new entry and bubble down
            self.entries[0] = (neighbor_idx, dot);
            self.bubble_down(0);
            true
        }

        #[inline]
        fn bubble_up(&mut self, mut idx: usize) {
            while idx > 0 {
                let parent = (idx - 1) / 2;
                if self.entries[idx].1 < self.entries[parent].1 {
                    self.entries.swap(idx, parent);
                    idx = parent;
                } else {
                    break;
                }
            }
        }

        #[inline]
        fn bubble_down(&mut self, mut idx: usize) {
            let count = self.count as usize;
            loop {
                let left = 2 * idx + 1;
                let right = 2 * idx + 2;
                let mut smallest = idx;

                if left < count && self.entries[left].1 < self.entries[smallest].1 {
                    smallest = left;
                }
                if right < count && self.entries[right].1 < self.entries[smallest].1 {
                    smallest = right;
                }

                if smallest == idx {
                    break;
                }

                self.entries.swap(idx, smallest);
                idx = smallest;
            }
        }

        /// Extract sorted neighbor indices (by distance, ascending = by dot, descending).
        fn to_sorted_indices(&self) -> Vec<usize> {
            let count = self.count as usize;
            let mut entries: Vec<(u32, f32)> = self.entries[..count].to_vec();
            // Sort by dot descending (higher dot = closer)
            entries.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            entries.iter().map(|(idx, _)| *idx as usize).collect()
        }
    }

    /// Per-point k-NN tracker using sorted array.
    /// Kept sorted by dot (descending), worst is always at position count-1.
    /// O(1) rejection, O(K) insert (but cache-friendly shifts).
    struct PointTopKSorted<const K: usize> {
        /// Sorted by dot descending: best at [0], worst at [count-1]
        entries: [(u32, f32); K],
        count: u8,
    }

    impl<const K: usize> PointTopKSorted<K> {
        fn new() -> Self {
            Self {
                entries: [(u32::MAX, f32::NEG_INFINITY); K],
                count: 0,
            }
        }

        /// Try to insert a neighbor - O(K) shift but cache-friendly
        #[inline]
        fn try_insert(&mut self, neighbor_idx: u32, dot: f32) -> bool {
            let count = self.count as usize;

            if count < K {
                // Not full - find insert position via linear search (small K)
                let mut pos = count;
                for i in 0..count {
                    if dot > self.entries[i].1 {
                        pos = i;
                        break;
                    }
                }

                // Shift right and insert
                if pos < count {
                    unsafe {
                        std::ptr::copy(
                            self.entries.as_ptr().add(pos),
                            self.entries.as_mut_ptr().add(pos + 1),
                            count - pos,
                        );
                    }
                }
                self.entries[pos] = (neighbor_idx, dot);
                self.count += 1;
                return true;
            }

            // Full - worst is at K-1, O(1) rejection
            if dot <= self.entries[K - 1].1 {
                return false;
            }

            // Find insert position
            let mut pos = K - 1;
            for i in 0..K - 1 {
                if dot > self.entries[i].1 {
                    pos = i;
                    break;
                }
            }

            // Shift right (dropping last) and insert
            if pos < K - 1 {
                unsafe {
                    std::ptr::copy(
                        self.entries.as_ptr().add(pos),
                        self.entries.as_mut_ptr().add(pos + 1),
                        K - 1 - pos,
                    );
                }
            }
            self.entries[pos] = (neighbor_idx, dot);
            true
        }

        /// Extract sorted neighbor indices - already sorted, just copy
        #[inline]
        fn to_sorted_indices(&self) -> Vec<usize> {
            let count = self.count as usize;
            self.entries[..count]
                .iter()
                .map(|(idx, _)| *idx as usize)
                .collect()
        }
    }

    /// Generate all unique cell pairs (A, B) where A <= B.
    /// Includes self-pairs (A, A) and cross-cell pairs with neighbors.
    fn generate_cell_pairs(grid: &CubeMapGrid) -> Vec<(usize, usize)> {
        let num_cells = 6 * grid.res * grid.res;
        let mut pairs = Vec::new();

        for cell in 0..num_cells {
            // Self-pair
            pairs.push((cell, cell));

            // Neighbor pairs where neighbor > cell (to avoid duplicates)
            let neighbors = grid.cell_neighbors(cell);
            let mut seen_neighbors: Vec<u32> = Vec::new();

            for &n in neighbors.iter() {
                if n != u32::MAX {
                    let n_usize = n as usize;
                    // Only add if n > cell (to ensure we process each pair once)
                    // and we haven't seen this neighbor yet (dedup cross-face corners)
                    if n_usize > cell && !seen_neighbors.contains(&n) {
                        pairs.push((cell, n_usize));
                        seen_neighbors.push(n);
                    }
                }
            }
        }

        pairs
    }

    /// Pair-based KNN with O(K) linear insert.
    fn pair_based_knn<const K: usize>(
        grid: &CubeMapGrid,
        points: &[Vec3],
    ) -> Vec<Vec<usize>> {
        let n = points.len();
        let mut topk: Vec<PointTopK<K>> = (0..n).map(|_| PointTopK::new()).collect();
        let pairs = generate_cell_pairs(grid);

        for (cell_a, cell_b) in pairs {
            let pts_a = grid.cell_points(cell_a);
            let pts_b = grid.cell_points(cell_b);

            if cell_a == cell_b {
                for (i_loc, &i) in pts_a.iter().enumerate() {
                    let pi = points[i as usize];
                    for &j in &pts_a[i_loc + 1..] {
                        let dot = pi.dot(points[j as usize]);
                        topk[i as usize].try_insert(j, dot);
                        topk[j as usize].try_insert(i, dot);
                    }
                }
            } else {
                for &i in pts_a {
                    let pi = points[i as usize];
                    for &j in pts_b {
                        let dot = pi.dot(points[j as usize]);
                        topk[i as usize].try_insert(j, dot);
                        topk[j as usize].try_insert(i, dot);
                    }
                }
            }
        }

        topk.iter().map(|t| t.to_sorted_indices()).collect()
    }

    /// Pair-based KNN with lazy worst_dot updates.
    fn pair_based_knn_lazy<const K: usize>(
        grid: &CubeMapGrid,
        points: &[Vec3],
    ) -> Vec<Vec<usize>> {
        let n = points.len();
        let mut topk: Vec<PointTopKLazy<K>> = (0..n).map(|_| PointTopKLazy::new()).collect();
        let pairs = generate_cell_pairs(grid);

        for (cell_a, cell_b) in pairs {
            let pts_a = grid.cell_points(cell_a);
            let pts_b = grid.cell_points(cell_b);

            if cell_a == cell_b {
                for (i_loc, &i) in pts_a.iter().enumerate() {
                    let pi = points[i as usize];
                    for &j in &pts_a[i_loc + 1..] {
                        let dot = pi.dot(points[j as usize]);
                        topk[i as usize].try_insert(j, dot);
                        topk[j as usize].try_insert(i, dot);
                    }
                }
            } else {
                for &i in pts_a {
                    let pi = points[i as usize];
                    for &j in pts_b {
                        let dot = pi.dot(points[j as usize]);
                        topk[i as usize].try_insert(j, dot);
                        topk[j as usize].try_insert(i, dot);
                    }
                }
            }
        }

        topk.iter().map(|t| t.to_sorted_indices()).collect()
    }

    /// Pair-based KNN with O(log K) heap insert.
    fn pair_based_knn_heap<const K: usize>(
        grid: &CubeMapGrid,
        points: &[Vec3],
    ) -> Vec<Vec<usize>> {
        let n = points.len();
        let mut topk: Vec<PointTopKHeap<K>> = (0..n).map(|_| PointTopKHeap::new()).collect();
        let pairs = generate_cell_pairs(grid);

        for (cell_a, cell_b) in pairs {
            let pts_a = grid.cell_points(cell_a);
            let pts_b = grid.cell_points(cell_b);

            if cell_a == cell_b {
                for (i_loc, &i) in pts_a.iter().enumerate() {
                    let pi = points[i as usize];
                    for &j in &pts_a[i_loc + 1..] {
                        let dot = pi.dot(points[j as usize]);
                        topk[i as usize].try_insert(j, dot);
                        topk[j as usize].try_insert(i, dot);
                    }
                }
            } else {
                for &i in pts_a {
                    let pi = points[i as usize];
                    for &j in pts_b {
                        let dot = pi.dot(points[j as usize]);
                        topk[i as usize].try_insert(j, dot);
                        topk[j as usize].try_insert(i, dot);
                    }
                }
            }
        }

        topk.iter().map(|t| t.to_sorted_indices()).collect()
    }

    /// Pair-based KNN with sorted array (O(K) insert, O(1) reject, no final sort).
    fn pair_based_knn_sorted<const K: usize>(
        grid: &CubeMapGrid,
        points: &[Vec3],
    ) -> Vec<Vec<usize>> {
        let n = points.len();
        let mut topk: Vec<PointTopKSorted<K>> = (0..n).map(|_| PointTopKSorted::new()).collect();
        let pairs = generate_cell_pairs(grid);

        for (cell_a, cell_b) in pairs {
            let pts_a = grid.cell_points(cell_a);
            let pts_b = grid.cell_points(cell_b);

            if cell_a == cell_b {
                for (i_loc, &i) in pts_a.iter().enumerate() {
                    let pi = points[i as usize];
                    for &j in &pts_a[i_loc + 1..] {
                        let dot = pi.dot(points[j as usize]);
                        topk[i as usize].try_insert(j, dot);
                        topk[j as usize].try_insert(i, dot);
                    }
                }
            } else {
                for &i in pts_a {
                    let pi = points[i as usize];
                    for &j in pts_b {
                        let dot = pi.dot(points[j as usize]);
                        topk[i as usize].try_insert(j, dot);
                        topk[j as usize].try_insert(i, dot);
                    }
                }
            }
        }

        topk.iter().map(|t| t.to_sorted_indices()).collect()
    }

    /// Pair-based KNN with per-bin topk storage for cache locality.
    ///
    /// Key ideas:
    /// 1. topk[bin][local_idx] instead of topk[global_idx]
    /// 2. Process pairs in spatial order: all pairs involving bin 0, then bin 1, etc.
    /// 3. When processing (bin_a, bin_b), bin_a's topk entries are hot in cache
    fn pair_based_knn_binned<const K: usize>(
        grid: &CubeMapGrid,
        points: &[Vec3],
    ) -> Vec<Vec<usize>> {
        let n = points.len();
        let num_cells = 6 * grid.res * grid.res;

        // Per-bin topk storage: topk[bin][local_idx]
        let mut bin_topk: Vec<Vec<PointTopKSorted<K>>> = (0..num_cells)
            .map(|cell| {
                let count = grid.cell_points(cell).len();
                (0..count).map(|_| PointTopKSorted::new()).collect()
            })
            .collect();

        // Process pairs in spatial order
        for cell_a in 0..num_cells {
            let pts_a = grid.cell_points(cell_a);
            if pts_a.is_empty() {
                continue;
            }

            // Self-pair: all (i, j) where i < j within cell_a
            for i_loc in 0..pts_a.len() {
                let i_global = pts_a[i_loc] as usize;
                let pi = points[i_global];

                for j_loc in (i_loc + 1)..pts_a.len() {
                    let j_global = pts_a[j_loc] as usize;
                    let dot = pi.dot(points[j_global]);

                    // Both updates are to bin_topk[cell_a] - cache friendly!
                    bin_topk[cell_a][i_loc].try_insert(pts_a[j_loc], dot);
                    bin_topk[cell_a][j_loc].try_insert(pts_a[i_loc], dot);
                }
            }

            // Cross-pairs with neighbors (only where neighbor > cell_a to avoid dupes)
            let neighbors = grid.cell_neighbors(cell_a);
            let mut seen_neighbors: Vec<u32> = Vec::new();

            for &neighbor in neighbors.iter() {
                if neighbor == u32::MAX {
                    continue;
                }
                let cell_b = neighbor as usize;

                // Skip if already processed or duplicate
                if cell_b <= cell_a || seen_neighbors.contains(&neighbor) {
                    continue;
                }
                seen_neighbors.push(neighbor);

                let pts_b = grid.cell_points(cell_b);

                for (i_loc, &i_global) in pts_a.iter().enumerate() {
                    let pi = points[i_global as usize];

                    for (j_loc, &j_global) in pts_b.iter().enumerate() {
                        let dot = pi.dot(points[j_global as usize]);

                        // Update cell_a's topk (hot!) and cell_b's topk
                        bin_topk[cell_a][i_loc].try_insert(j_global, dot);
                        bin_topk[cell_b][j_loc].try_insert(i_global, dot);
                    }
                }
            }
        }

        // Gather results back to global indices
        let mut results = vec![Vec::new(); n];
        for cell in 0..num_cells {
            let pts = grid.cell_points(cell);
            for (local_idx, &global_idx) in pts.iter().enumerate() {
                results[global_idx as usize] = bin_topk[cell][local_idx].to_sorted_indices();
            }
        }

        results
    }

    #[test]
    #[ignore] // cargo test pair_based_knn_benchmark --release -- --ignored --nocapture
    fn pair_based_knn_benchmark() {
        const K: usize = 24;
        let n = 100_000;
        let points = fibonacci_sphere_points(n);
        let points_a = to_vec3a(&points);

        println!("\n=== KNN Benchmark (n={}, k={}) ===", n, K);

        // Try different densities
        let densities = [12.0, 16.0, 20.0, 24.0, 32.0];

        println!("\nDensity sweep:");
        println!("{:>8} {:>4} {:>7} {:>7} {:>7} {:>7}",
            "density", "res", "base", "simd3x3", "tiled5x5", "speedup");

        let mut best_baseline = (f64::MAX, 0.0);
        let mut best_simd = (f64::MAX, 0.0);
        let mut best_tiled = (f64::MAX, 0.0);

        for &target_pts in &densities {
            let res = ((n as f64 / (6.0 * target_pts)).sqrt() as usize).max(4);
            let grid = CubeMapGrid::new(&points, res);

            // Warm up
            let _ = baseline_knn_all(&grid, &points_a, K);
            let _ = batched_knn_simd(&grid, &points, K);
            let _ = batched_knn_tiled_5x5(&grid, &points, K);

            let t0 = Instant::now();
            let _ = baseline_knn_all(&grid, &points_a, K);
            let baseline_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

            let t0 = Instant::now();
            let _ = batched_knn_simd(&grid, &points, K);
            let simd_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

            let t0 = Instant::now();
            let _ = batched_knn_tiled_5x5(&grid, &points, K);
            let tiled_us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;

            let tiled_speedup = baseline_us / tiled_us;

            println!("{:>8.0} {:>4} {:>7.2} {:>7.2} {:>7.2} {:>7.2}x",
                target_pts, res, baseline_us, simd_us, tiled_us, tiled_speedup);

            if baseline_us < best_baseline.0 {
                best_baseline = (baseline_us, target_pts);
            }
            if simd_us < best_simd.0 {
                best_simd = (simd_us, target_pts);
            }
            if tiled_us < best_tiled.0 {
                best_tiled = (tiled_us, target_pts);
            }
        }

        println!("\n=== Best times (µs/query, each at optimal density) ===");
        println!("Baseline (heap):      {:.2}µs @ density={:.0}", best_baseline.0, best_baseline.1);
        println!("SIMD 3x3:             {:.2}µs @ density={:.0}", best_simd.0, best_simd.1);
        println!("Tiled 5x5:            {:.2}µs @ density={:.0}", best_tiled.0, best_tiled.1);

        println!("\n=== Speedups (vs baseline) ===");
        println!("SIMD 3x3:             {:.2}x", best_baseline.0 / best_simd.0);
        println!("Tiled 5x5:            {:.2}x", best_baseline.0 / best_tiled.0);

        // Verify correctness for tiled 5x5
        let res = ((n as f64 / (6.0 * best_tiled.1)).sqrt() as usize).max(4);
        let grid = CubeMapGrid::new(&points, res);

        println!("\n=== Correctness check (tiled 5x5 @ density={:.0}) ===", best_tiled.1);

        let results = batched_knn_tiled_5x5(&grid, &points, K);

        // Check against brute force (with tie tolerance)
        let mut errors = 0;
        let mut tie_diffs = 0;
        let tie_tolerance = 1e-5;
        let sample_step = n / 100;

        for i in (0..n).step_by(sample_step) {
            let expected = brute_force_knn(&points, i, K);
            let got = &results[i];

            let expected_set: std::collections::HashSet<_> = expected.iter().collect();
            let got_set: std::collections::HashSet<_> = got.iter().collect();

            if expected_set != got_set {
                let query = points[i];
                let d_k = if got.len() == K {
                    (query - points[got[K - 1]]).length()
                } else {
                    f32::MAX
                };

                // Check if difference is just tie-breaking
                let is_tie = expected_set.difference(&got_set).all(|&&m| {
                    let d = (query - points[m]).length();
                    (d - d_k).abs() < tie_tolerance
                });

                if is_tie {
                    tie_diffs += 1;
                } else {
                    errors += 1;
                    if errors <= 3 {
                        let missing: Vec<_> = expected_set.difference(&got_set).map(|&&x| x).collect();
                        println!("  ERROR at {}: missing {:?}", i, missing);
                        for &m in &missing {
                            let d_m = (query - points[m]).length();
                            let m_cell = grid.point_to_cell(points[m]);
                            println!("    {} at dist={:.6}, cell={}", m, d_m, m_cell);
                        }
                    }
                }
            }
        }

        println!("Sampled {} points: {} hard errors, {} tie diffs", n / sample_step, errors, tie_diffs);

        // Dot count comparison
        let stats = grid.stats();
        let simd_3x3_dots = 9.0 * stats.avg_points_per_cell;
        let tiled_5x5_dots = 25.0 * stats.avg_points_per_cell;
        println!("\n=== Dot counts per query ===");
        println!("SIMD 3x3:   ~{:.0} dots", simd_3x3_dots);
        println!("Tiled 5x5:  ~{:.0} dots ({:.1}x more)", tiled_5x5_dots, tiled_5x5_dots / simd_3x3_dots);
    }
}
