//! Cube-map based spatial grid for fast k-NN on unit sphere.
//!
//! Projects sphere onto 6 cube faces, divides each into a regular grid.
//! O(n) build, O(1) cell lookup.
//!
//! Queries use a best-first expansion over neighboring cells with a conservative
//! distance lower bound per cell. Typical uniform inputs terminate after
//! visiting a handful of cells; worst-case can fall back to brute force.
//!
//! Supports resumable queries: start with small k, expand to larger k without
//! re-doing work. Uses a fixed-size candidate buffer (MAX_K) for cache efficiency.

use glam::Vec3;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Status of a k-NN query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnStatus {
    /// More neighbors may be available; query can be resumed with a larger k.
    CanResume,
    /// Search exhausted; no more neighbors available beyond what was returned.
    Exhausted,
}

impl KnnStatus {
    /// Returns true if the search is exhausted.
    #[inline]
    pub fn is_exhausted(self) -> bool {
        self == KnnStatus::Exhausted
    }
}

/// Maximum neighbors we'll ever track. Brute-force fallback fills this entire buffer.
pub const MAX_K: usize = 48;

/// A f32 wrapper that implements Ord using total_cmp.
/// Unlike NotNan, this doesn't check for NaN - it just orders NaN consistently.
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF32(f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32 {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl OrdF32 {
    #[inline]
    fn new(v: f32) -> Self {
        OrdF32(v)
    }

    #[inline]
    fn get(self) -> f32 {
        self.0
    }
}

/// Cube-map spatial grid for points on unit sphere.
pub struct CubeMapGrid {
    res: usize,
    /// Start index into point_indices for each cell, plus final length.
    /// Length: 6 * res² + 1
    cell_offsets: Vec<u32>,
    /// Point indices grouped by cell.
    /// Length: n (number of points)
    point_indices: Vec<u32>,
    /// Precomputed 3×3 neighborhood for each cell.
    /// 9 entries per cell (self + 8 neighbors), u32::MAX = invalid.
    /// Length: 6 * res² * 9
    neighbors: Vec<u32>,
    /// Unit vector at the center of each cell (on the sphere).
    cell_centers: Vec<Vec3>,
    /// Spherical cap radius around `cell_centers[cell]` that conservatively contains the cell.
    /// Stored as cos/sin for fast per-query bounds.
    cell_cos_radius: Vec<f32>,
    cell_sin_radius: Vec<f32>,
}

/// Map a point on unit sphere to (face, u, v) where u,v ∈ [-1, 1].
#[inline]
fn point_to_face_uv(p: Vec3) -> (usize, f32, f32) {
    let (x, y, z) = (p.x, p.y, p.z);
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());

    if ax >= ay && ax >= az {
        // ±X
        if x >= 0.0 {
            (0, -z / ax, y / ax)
        } else {
            (1, z / ax, y / ax)
        }
    } else if ay >= ax && ay >= az {
        // ±Y
        if y >= 0.0 {
            (2, x / ay, -z / ay)
        } else {
            (3, x / ay, z / ay)
        }
    } else {
        // ±Z
        if z >= 0.0 {
            (4, x / az, y / az)
        } else {
            (5, -x / az, y / az)
        }
    }
}

/// Convert (face, u, v) to cell index.
#[inline]
fn face_uv_to_cell(face: usize, u: f32, v: f32, res: usize) -> usize {
    // Map [-1, 1] -> [0, res)
    let fu = ((u + 1.0) * 0.5) * (res as f32);
    let fv = ((v + 1.0) * 0.5) * (res as f32);
    let iu = (fu as usize).min(res - 1);
    let iv = (fv as usize).min(res - 1);
    face * res * res + iv * res + iu
}

/// Convert (face, u, v) back to a 3D point (inverse of point_to_face_uv).
#[inline]
fn face_uv_to_3d(face: usize, u: f32, v: f32) -> Vec3 {
    // Project onto cube face, then normalize to sphere
    let p = match face {
        0 => Vec3::new(1.0, v, -u),      // +X: u = -z/x, v = y/x
        1 => Vec3::new(-1.0, v, u),      // -X: u = z/|x|, v = y/|x|
        2 => Vec3::new(u, 1.0, -v),      // +Y: u = x/y, v = -z/y
        3 => Vec3::new(u, -1.0, v),      // -Y: u = x/|y|, v = z/|y|
        4 => Vec3::new(u, v, 1.0),       // +Z: u = x/z, v = y/z
        5 => Vec3::new(-u, v, -1.0),     // -Z: u = -x/|z|, v = y/|z|
        _ => unreachable!(),
    };
    p.normalize()
}

/// Convert cell index to (face, iu, iv).
#[inline]
fn cell_to_face_ij(cell: usize, res: usize) -> (usize, usize, usize) {
    let face = cell / (res * res);
    let rem = cell % (res * res);
    let iv = rem / res;
    let iu = rem % res;
    (face, iu, iv)
}

/// Reusable per-query scratch buffers.
///
/// Uses a fixed-size sorted array for candidates (instead of a heap) for:
/// - O(1) access to k-th distance for pruning
/// - Cache-friendly linear scans for small k
/// - Zero-cost resume (just slice the existing buffer)
///
/// For performance (especially parallel queries), prefer `CubeMapGrid::make_scratch()` and
/// `CubeMapGrid::find_k_nearest_with_scratch()`.
pub struct CubeMapGridScratch {
    visited_stamp: Vec<u32>,
    stamp: u32,
    cell_heap: BinaryHeap<Reverse<(OrdF32, u32)>>,
    /// Fixed-size candidate buffer, sorted by distance (ascending).
    /// (dist_sq, point_idx)
    candidates: [(f32, u32); MAX_K],
    /// Number of valid candidates in the buffer.
    candidate_count: usize,
    /// If true, we've done a brute-force scan and have all MAX_K neighbors.
    /// Future queries just slice the buffer - no more grid expansion needed.
    exhausted: bool,
    /// Track limit for resumable queries (set by find_k_nearest_resumable_into).
    /// Used by resume_k_nearest_into to enforce consistent tracking.
    track_limit: usize,
}

impl CubeMapGridScratch {
    pub fn new(num_cells: usize) -> Self {
        Self {
            visited_stamp: vec![0; num_cells],
            stamp: 0,
            cell_heap: BinaryHeap::new(),
            candidates: [(f32::INFINITY, u32::MAX); MAX_K],
            candidate_count: 0,
            exhausted: false,
            track_limit: MAX_K,
        }
    }

    #[inline]
    fn begin_query(&mut self) {
        self.cell_heap.clear();
        self.candidate_count = 0;
        self.exhausted = false;
        self.track_limit = MAX_K;
        // Reset candidates to infinity
        for c in &mut self.candidates {
            *c = (f32::INFINITY, u32::MAX);
        }

        // Stamp 0 means "unvisited". Avoid ever using stamp 0 for a query.
        self.stamp = self.stamp.wrapping_add(1).max(1);
        if self.stamp == u32::MAX {
            self.visited_stamp.fill(0);
            self.stamp = 1;
        }
    }

    #[inline]
    fn mark_visited(&mut self, cell: u32) -> bool {
        let idx = cell as usize;
        if self.visited_stamp[idx] == self.stamp {
            return false;
        }
        self.visited_stamp[idx] = self.stamp;
        true
    }

    #[inline]
    fn push_cell(&mut self, cell: u32, bound_dist_sq: f32) {
        self.cell_heap.push(Reverse((OrdF32::new(bound_dist_sq), cell)));
    }

    #[inline]
    fn peek_cell(&self) -> Option<(f32, u32)> {
        self.cell_heap.peek().map(|Reverse((bound, cell))| (bound.get(), *cell))
    }

    #[inline]
    fn pop_cell(&mut self) -> Option<(f32, u32)> {
        self.cell_heap.pop().map(|Reverse((bound, cell))| (bound.get(), cell))
    }

    /// Get the distance to the k-th candidate (for pruning).
    /// Returns infinity if we have fewer than k candidates.
    #[inline]
    fn kth_dist_sq(&self, k: usize) -> f32 {
        if k == 0 || k > MAX_K {
            return f32::INFINITY;
        }
        self.candidates[k - 1].0
    }

    /// Try to add a neighbor, tracking up to `limit` candidates.
    /// Maintains sorted order (ascending by distance).
    #[inline]
    fn try_add_neighbor_limit(&mut self, idx: usize, dist_sq: f32, limit: usize) {
        let limit = limit.min(MAX_K);
        if dist_sq.is_nan() || (self.candidate_count >= limit && dist_sq >= self.candidates[limit - 1].0) {
            return;
        }

        let idx_u32 = idx as u32;

        // Find insertion point (binary search for sorted insert)
        let search_end = self.candidate_count.min(limit);
        let insert_pos = self.candidates[..search_end]
            .partition_point(|&(d, _)| d < dist_sq);

        // Check for duplicate
        if insert_pos < self.candidate_count && self.candidates[insert_pos].1 == idx_u32 {
            return;
        }

        // Shift elements right and insert
        if insert_pos < limit {
            let shift_end = self.candidate_count.min(limit - 1);
            // Shift right (drop last if at limit)
            for i in (insert_pos..shift_end).rev() {
                self.candidates[i + 1] = self.candidates[i];
            }
            self.candidates[insert_pos] = (dist_sq, idx_u32);
            if self.candidate_count < limit {
                self.candidate_count += 1;
            }
        }
    }

    /// Try to add a neighbor, tracking up to MAX_K candidates (for brute-force fill).
    #[inline]
    fn try_add_neighbor(&mut self, idx: usize, dist_sq: f32) {
        self.try_add_neighbor_limit(idx, dist_sq, MAX_K);
    }

    /// Copy the first k candidate indices into output vec (sorted by distance).
    fn copy_k_indices_into(&self, k: usize, out: &mut Vec<usize>) {
        out.clear();
        let count = k.min(self.candidate_count);
        out.reserve(count);
        for i in 0..count {
            out.push(self.candidates[i].1 as usize);
        }
    }
}

impl CubeMapGrid {
    /// Build a cube-map grid from points on unit sphere.
    ///
    /// `res` controls grid resolution: 6 * res² total cells.
    /// For n points, good choices:
    /// - res ≈ sqrt(n / 300) for ~50 points per cell
    /// - res ≈ sqrt(n / 600) for ~100 points per cell
    pub fn new(points: &[Vec3], res: usize) -> Self {
        assert!(res > 0, "CubeMapGrid requires res > 0");
        let num_cells = 6 * res * res;

        // Step 1: Count points per cell
        let mut cell_counts = vec![0u32; num_cells];
        for p in points {
            let (face, u, v) = point_to_face_uv(*p);
            let cell = face_uv_to_cell(face, u, v, res);
            cell_counts[cell] += 1;
        }

        // Step 2: Prefix sum to get offsets
        let mut cell_offsets = Vec::with_capacity(num_cells + 1);
        cell_offsets.push(0);
        let mut sum = 0u32;
        for &count in &cell_counts {
            sum += count;
            cell_offsets.push(sum);
        }

        // Step 3: Scatter points into cells
        let mut point_indices = vec![0u32; points.len()];
        let mut cell_cursors = cell_offsets[..num_cells].to_vec();
        for (i, p) in points.iter().enumerate() {
            let (face, u, v) = point_to_face_uv(*p);
            let cell = face_uv_to_cell(face, u, v, res);
            let pos = cell_cursors[cell] as usize;
            point_indices[pos] = i as u32;
            cell_cursors[cell] += 1;
        }

        // Step 4: Precompute neighbors for each cell
        let neighbors = Self::compute_all_neighbors(res);
        let (cell_centers, cell_cos_radius, cell_sin_radius) = Self::compute_cell_bounds(res);

        CubeMapGrid {
            res,
            cell_offsets,
            point_indices,
            neighbors,
            cell_centers,
            cell_cos_radius,
            cell_sin_radius,
        }
    }

    /// Compute 3×3 neighborhood for all cells.
    fn compute_all_neighbors(res: usize) -> Vec<u32> {
        let num_cells = 6 * res * res;
        let mut neighbors = vec![u32::MAX; num_cells * 9];

        for cell in 0..num_cells {
            let (face, iu, iv) = cell_to_face_ij(cell, res);
            let base = cell * 9;
            let mut idx = 0;

            for dv in [-1i32, 0, 1] {
                for du in [-1i32, 0, 1] {
                    let niu = iu as i32 + du;
                    let niv = iv as i32 + dv;

                    let neighbor_cell = if niu >= 0 && niu < res as i32 && niv >= 0 && niv < res as i32 {
                        // Same face
                        Some(face * res * res + (niv as usize) * res + (niu as usize))
                    } else {
                        // Cross face boundary using 3D projection
                        Self::get_cross_face_neighbor(face, niu, niv, res)
                    };

                    neighbors[base + idx] = neighbor_cell.map(|c| c as u32).unwrap_or(u32::MAX);
                    idx += 1;
                }
            }
        }

        neighbors
    }

    /// Get neighbor cell when crossing a face boundary.
    /// Uses 3D coordinate conversion to find the correct neighbor cell.
    fn get_cross_face_neighbor(face: usize, niu: i32, niv: i32, res: usize) -> Option<usize> {
        // Clamp coordinates to slightly outside the face (for edge/corner crossing)
        // For corners, this will project to a position near the cube corner
        let niu_clamped = niu.clamp(-1, res as i32);
        let niv_clamped = niv.clamp(-1, res as i32);

        // Convert to UV coordinates (may be slightly outside [-1, 1])
        let u = (niu_clamped as f32 + 0.5) / res as f32 * 2.0 - 1.0;
        let v = (niv_clamped as f32 + 0.5) / res as f32 * 2.0 - 1.0;

        // Convert to 3D point on the cube face, then normalize to sphere
        let point_3d = face_uv_to_3d(face, u, v);

        // Use point_to_face_uv to find which face/cell this maps to
        let (new_face, new_u, new_v) = point_to_face_uv(point_3d);

        // Convert to cell coordinates
        let new_iu = (((new_u + 1.0) * 0.5) * res as f32) as usize;
        let new_iv = (((new_v + 1.0) * 0.5) * res as f32) as usize;

        // Clamp to valid range
        let new_iu = new_iu.min(res - 1);
        let new_iv = new_iv.min(res - 1);

        Some(new_face * res * res + new_iv * res + new_iu)
    }

    /// Get cell index for a point.
    #[inline]
    pub fn point_to_cell(&self, p: Vec3) -> usize {
        let (face, u, v) = point_to_face_uv(p);
        face_uv_to_cell(face, u, v, self.res)
    }

    /// Get points in a cell.
    #[inline]
    pub fn cell_points(&self, cell: usize) -> &[u32] {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;
        &self.point_indices[start..end]
    }

    /// Get the 9 neighbor cells (including self) for a cell.
    #[inline]
    pub fn cell_neighbors(&self, cell: usize) -> &[u32; 9] {
        let base = cell * 9;
        self.neighbors[base..base + 9].try_into().unwrap()
    }

    /// Create a reusable scratch buffer for fast repeated queries.
    pub fn make_scratch(&self) -> CubeMapGridScratch {
        CubeMapGridScratch::new(6 * self.res * self.res)
    }

    /// Conservative lower bound on squared Euclidean distance from `query` to any point in `cell`.
    ///
    /// Uses a spherical cap that contains the cell and triangle inequality on the sphere.
    #[inline]
    fn cell_min_dist_sq(&self, query: Vec3, cell: usize) -> f32 {
        let center = self.cell_centers[cell];
        let mut cos_d = query.dot(center);
        cos_d = cos_d.clamp(-1.0, 1.0);

        let cos_r = self.cell_cos_radius[cell];
        let sin_r = self.cell_sin_radius[cell];

        // If the query direction is within the cell's cap, the minimum distance can be 0.
        if cos_d > cos_r {
            return 0.0;
        }

        // cos(d - r) = cos d cos r + sin d sin r
        let sin_d = (1.0 - cos_d * cos_d).max(0.0).sqrt();
        let max_dot_upper = (cos_d * cos_r + sin_d * sin_r).clamp(-1.0, 1.0);
        2.0 - 2.0 * max_dot_upper
    }

    /// Find k nearest neighbors for a point.
    /// Returns indices sorted by distance (closest first).
    ///
    /// For high-throughput usage, prefer `find_k_nearest_with_scratch`.
    pub fn find_k_nearest(&self, points: &[Vec3], query: Vec3, query_idx: usize, k: usize) -> Vec<usize> {
        let mut scratch = self.make_scratch();
        self.find_k_nearest_with_scratch(points, query, query_idx, k, &mut scratch)
    }

    /// Scratch-based k-NN query that writes results into `out_indices` (sorted closest-first).
    ///
    /// This is the preferred high-throughput API: it avoids per-query allocations.
    pub fn find_k_nearest_with_scratch_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) {
        let n = points.len();
        out_indices.clear();

        if k == 0 || n <= 1 {
            return;
        }
        let k = k.min(n - 1).min(MAX_K);

        let num_cells = 6 * self.res * self.res;
        scratch.begin_query();

        let start_cell = self.point_to_cell(query) as u32;
        scratch.mark_visited(start_cell);
        scratch.push_cell(start_cell, 0.0);

        // If the search balloons, brute force is usually cheaper than touching most cells
        // (especially when many cells are empty).
        let max_cells_before_bruteforce = (num_cells / 2).max(64);
        let mut visited_cells = 0usize;

        while let Some((bound_dist_sq, cell_u32)) = scratch.pop_cell() {
            // Prune: if next cell's bound >= k-th best distance, we're done
            let kth_dist = scratch.kth_dist_sq(k);
            if scratch.candidate_count >= k && bound_dist_sq >= kth_dist {
                break;
            }

            visited_cells += 1;
            if visited_cells > max_cells_before_bruteforce {
                self.bruteforce_fill_max_k(points, query, query_idx, scratch);
                scratch.copy_k_indices_into(k, out_indices);
                return;
            }

            let cell = cell_u32 as usize;
            debug_assert!(cell < num_cells);

            let start = self.cell_offsets[cell] as usize;
            let end = self.cell_offsets[cell + 1] as usize;
            for &pidx_u32 in &self.point_indices[start..end] {
                let pidx = pidx_u32 as usize;
                if pidx == query_idx {
                    continue;
                }
                let dist_sq = (points[pidx] - query).length_squared();
                scratch.try_add_neighbor_limit(pidx, dist_sq, k);
            }

            // Neighbor expansion: use the precomputed 3×3 neighborhood as an 8-neighbor graph.
            let base = cell * 9;
            for &ncell in &self.neighbors[base..base + 9] {
                if ncell == u32::MAX || ncell == cell_u32 {
                    continue;
                }
                if !scratch.mark_visited(ncell) {
                    continue;
                }
                let nb = self.cell_min_dist_sq(query, ncell as usize);
                scratch.push_cell(ncell, nb);
            }
        }

        if scratch.candidate_count < k {
            self.bruteforce_fill_max_k(points, query, query_idx, scratch);
        }

        scratch.copy_k_indices_into(k, out_indices);
    }

    /// Scratch-based k-NN query (best-first over neighboring cells).
    ///
    /// Guarantees returning `min(k, n-1)` indices (excluding `query_idx`), falling back to brute
    /// force if the cell expansion becomes too broad.
    pub fn find_k_nearest_with_scratch(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        scratch: &mut CubeMapGridScratch,
    ) -> Vec<usize> {
        let mut out = Vec::with_capacity(k);
        self.find_k_nearest_with_scratch_into(points, query, query_idx, k, scratch, &mut out);
        out
    }

    /// Start a resumable k-NN query.
    ///
    /// Unlike `find_k_nearest_with_scratch_into`, this preserves scratch state so the query
    /// can be resumed with `resume_k_nearest_into` to fetch additional neighbors.
    ///
    /// `track_limit` controls how many candidates we track internally. This bounds
    /// how far we can resume: if track_limit=24, we can resume up to k=24 without
    /// losing any neighbors. For k > track_limit, a fresh query would be needed.
    ///
    /// The track_limit is stored in the scratch for use by `resume_k_nearest_into`.
    pub fn find_k_nearest_resumable_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        let n = points.len();
        out_indices.clear();

        if k == 0 || n <= 1 {
            return KnnStatus::Exhausted;
        }
        let k = k.min(n - 1).min(MAX_K);
        let track_limit = track_limit.max(k).min(MAX_K);

        let num_cells = 6 * self.res * self.res;
        scratch.begin_query();
        scratch.track_limit = track_limit;

        let start_cell = self.point_to_cell(query) as u32;
        scratch.mark_visited(start_cell);
        scratch.push_cell(start_cell, 0.0);

        let exhausted = self.knn_search_loop_resumable(points, query, query_idx, k, track_limit, num_cells, scratch);

        scratch.copy_k_indices_into(k, out_indices);
        if exhausted { KnnStatus::Exhausted } else { KnnStatus::CanResume }
    }

    /// Resume a k-NN query to fetch additional neighbors.
    ///
    /// Call this after `find_k_nearest_resumable_into` when you need more neighbors.
    /// `new_k` should be larger than the previous k value but within the original
    /// `track_limit` (stored in scratch). If new_k > track_limit, results may be incomplete.
    pub fn resume_k_nearest_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        new_k: usize,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        let n = points.len();
        out_indices.clear();

        if new_k == 0 || n <= 1 {
            return KnnStatus::Exhausted;
        }
        let new_k = new_k.min(n - 1).min(MAX_K);
        let track_limit = scratch.track_limit.max(new_k).min(MAX_K);

        // If already exhausted (brute force done), just slice the buffer
        if scratch.exhausted {
            scratch.copy_k_indices_into(new_k, out_indices);
            return KnnStatus::Exhausted;
        }

        // If we already have enough candidates, just return them
        if scratch.candidate_count >= new_k {
            let kth_dist = scratch.kth_dist_sq(new_k);
            // Check if the next cell (if any) could have closer points
            if let Some((bound, _)) = scratch.peek_cell() {
                if bound >= kth_dist {
                    scratch.copy_k_indices_into(new_k, out_indices);
                    return KnnStatus::CanResume; // Not exhausted, but have enough
                }
            } else {
                // No more cells to expand
                scratch.copy_k_indices_into(new_k, out_indices);
                return KnnStatus::Exhausted;
            }
        }

        let num_cells = 6 * self.res * self.res;
        let exhausted = self.knn_search_loop_resumable(points, query, query_idx, new_k, track_limit, num_cells, scratch);

        scratch.copy_k_indices_into(new_k, out_indices);
        if exhausted { KnnStatus::Exhausted } else { KnnStatus::CanResume }
    }

    /// Core search loop for resumable queries.
    /// Uses peek-before-pop so we can pause without losing cells.
    ///
    /// `track_limit` controls how many candidates we track. This bounds how far
    /// we can resume: if track_limit=24, we can resume up to k=24 but not beyond.
    ///
    /// Returns `true` if exhausted (no more cells to expand or brute force done).
    fn knn_search_loop_resumable(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        num_cells: usize,
        scratch: &mut CubeMapGridScratch,
    ) -> bool {
        let max_cells_before_bruteforce = (num_cells / 2).max(64);
        let mut visited_cells = 0usize;

        loop {
            // Peek first to check if we should stop (without losing the cell)
            let Some((bound_dist_sq, cell_u32)) = scratch.peek_cell() else {
                return true; // No more cells - exhausted
            };

            // Prune: if next cell's bound >= k-th best distance, we can stop
            let kth_dist = scratch.kth_dist_sq(k);
            if scratch.candidate_count >= k && bound_dist_sq >= kth_dist {
                return false; // Not exhausted, can resume with larger k
            }

            // Now actually pop the cell
            scratch.pop_cell();

            visited_cells += 1;
            if visited_cells > max_cells_before_bruteforce {
                // Fall back to brute force - fills MAX_K and marks exhausted
                self.bruteforce_fill_max_k(points, query, query_idx, scratch);
                return true;
            }

            let cell = cell_u32 as usize;
            debug_assert!(cell < num_cells);

            let start = self.cell_offsets[cell] as usize;
            let end = self.cell_offsets[cell + 1] as usize;
            for &pidx_u32 in &self.point_indices[start..end] {
                let pidx = pidx_u32 as usize;
                if pidx == query_idx {
                    continue;
                }
                let dist_sq = (points[pidx] - query).length_squared();
                // Track up to track_limit so we can resume within that bound
                scratch.try_add_neighbor_limit(pidx, dist_sq, track_limit);
            }

            // Neighbor expansion
            let base = cell * 9;
            for &ncell in &self.neighbors[base..base + 9] {
                if ncell == u32::MAX || ncell == cell_u32 {
                    continue;
                }
                if !scratch.mark_visited(ncell) {
                    continue;
                }
                let nb = self.cell_min_dist_sq(query, ncell as usize);
                scratch.push_cell(ncell, nb);
            }
        }
    }

    /// Brute-force scan that fills the entire MAX_K buffer and marks exhausted.
    /// After this, any resume just slices the existing buffer.
    fn bruteforce_fill_max_k(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        scratch: &mut CubeMapGridScratch,
    ) {
        for (idx, p) in points.iter().enumerate() {
            if idx == query_idx {
                continue;
            }
            let dist_sq = (*p - query).length_squared();
            scratch.try_add_neighbor(idx, dist_sq);
        }
        scratch.exhausted = true;
    }

    fn compute_cell_bounds(res: usize) -> (Vec<Vec3>, Vec<f32>, Vec<f32>) {
        let num_cells = 6 * res * res;
        let mut centers = Vec::with_capacity(num_cells);
        let mut cos_r = Vec::with_capacity(num_cells);
        let mut sin_r = Vec::with_capacity(num_cells);

        // 5×5 samples per cell to conservatively bound the spherical cap.
        const TS: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];

        for cell in 0..num_cells {
            let (face, iu, iv) = cell_to_face_ij(cell, res);

            let u0 = (iu as f32) / res as f32 * 2.0 - 1.0;
            let u1 = ((iu + 1) as f32) / res as f32 * 2.0 - 1.0;
            let v0 = (iv as f32) / res as f32 * 2.0 - 1.0;
            let v1 = ((iv + 1) as f32) / res as f32 * 2.0 - 1.0;

            let uc = (u0 + u1) * 0.5;
            let vc = (v0 + v1) * 0.5;
            let center = face_uv_to_3d(face, uc, vc);

            let mut max_angle = 0.0f32;
            for &tv in &TS {
                let v = v0 + (v1 - v0) * tv;
                for &tu in &TS {
                    let u = u0 + (u1 - u0) * tu;
                    let p = face_uv_to_3d(face, u, v);
                    let dot = center.dot(p).clamp(-1.0, 1.0);
                    let angle = dot.acos();
                    max_angle = max_angle.max(angle);
                }
            }

            // Inflate slightly to avoid underestimation from float error / sampling.
            let radius = (max_angle + 1e-4).min(std::f32::consts::PI);
            centers.push(center);
            cos_r.push(radius.cos());
            sin_r.push(radius.sin());
        }

        (centers, cos_r, sin_r)
    }

    /// Statistics about the grid.
    pub fn stats(&self) -> GridStats {
        let num_cells = 6 * self.res * self.res;
        let mut min_pts = u32::MAX;
        let mut max_pts = 0u32;
        let mut empty = 0usize;

        for cell in 0..num_cells {
            let count = self.cell_offsets[cell + 1] - self.cell_offsets[cell];
            min_pts = min_pts.min(count);
            max_pts = max_pts.max(count);
            if count == 0 {
                empty += 1;
            }
        }

        GridStats {
            num_cells,
            num_points: self.point_indices.len(),
            min_points_per_cell: min_pts as usize,
            max_points_per_cell: max_pts as usize,
            empty_cells: empty,
            avg_points_per_cell: self.point_indices.len() as f64 / num_cells as f64,
        }
    }
}

#[derive(Debug)]
pub struct GridStats {
    pub num_cells: usize,
    pub num_points: usize,
    pub min_points_per_cell: usize,
    pub max_points_per_cell: usize,
    pub empty_cells: usize,
    pub avg_points_per_cell: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::gpu_voronoi::{build_kdtree, find_k_nearest as kiddo_find_k_nearest};
    use crate::geometry::fibonacci_sphere_points_with_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn generate_test_points(n: usize, seed: u64) -> Vec<Vec3> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mean_spacing = (4.0 * std::f32::consts::PI / n as f32).sqrt();
        let jitter = mean_spacing * 0.25;
        fibonacci_sphere_points_with_rng(n, jitter, &mut rng)
    }

    #[test]
    fn test_cube_grid_basic() {
        let points = generate_test_points(10_000, 12345);
        let grid = CubeMapGrid::new(&points, 20);

        let stats = grid.stats();
        println!("Grid stats: {:?}", stats);

        // Check that all points are accounted for
        assert_eq!(stats.num_points, 10_000);

        // Check reasonable distribution
        assert!(stats.avg_points_per_cell > 1.0);
        assert!(stats.max_points_per_cell < 500); // Not too uneven
    }

    #[test]
    fn test_cube_grid_cell_bounds_are_conservative() {
        // Sanity check: random points inside some cells should be within the precomputed cap.
        // This doesn't prove correctness but guards against gross underestimation.
        use rand::Rng;

        let points = generate_test_points(10_000, 12345);
        let grid = CubeMapGrid::new(&points, 32);
        let mut rng = rand::thread_rng();

        let num_cells = 6 * grid.res * grid.res;
        for _ in 0..100 {
            let cell = rng.gen_range(0..num_cells);
            let (face, iu, iv) = cell_to_face_ij(cell, grid.res);
            let u0 = (iu as f32) / grid.res as f32 * 2.0 - 1.0;
            let u1 = ((iu + 1) as f32) / grid.res as f32 * 2.0 - 1.0;
            let v0 = (iv as f32) / grid.res as f32 * 2.0 - 1.0;
            let v1 = ((iv + 1) as f32) / grid.res as f32 * 2.0 - 1.0;

            let center = grid.cell_centers[cell];
            let cap_angle = grid.cell_cos_radius[cell].clamp(-1.0, 1.0).acos() + 1e-3;

            for _ in 0..100 {
                let u = rng.gen_range(u0..u1);
                let v = rng.gen_range(v0..v1);
                let p = face_uv_to_3d(face, u, v);
                let ang = center.dot(p).clamp(-1.0, 1.0).acos();
                assert!(ang <= cap_angle, "cell cap underestimates (ang={ang}, cap={cap_angle})");
            }
        }
    }

    #[test]
    #[ignore] // Run with: cargo test test_cube_grid_exhaustive -- --ignored --nocapture
    fn test_cube_grid_exhaustive() {
        use std::collections::HashSet;

        println!("\n=== Exhaustive CubeMapGrid Correctness Test ===\n");

        for &n in &[10_000usize, 50_000] {  // Skip small n where resolution is problematic
            let points = generate_test_points(n, 12345);
            let k = 24;
            let res = ((n as f64 / 300.0).sqrt() as usize).max(4);

            let grid = CubeMapGrid::new(&points, res);
            let mut scratch = grid.make_scratch();
            let (tree, entries) = build_kdtree(&points);

            let mut exact_match = 0;
            let mut missing_neighbors = 0;
            let mut wrong_neighbors = 0;
            let mut worst_overlap = k;

            let mut first_failure_printed = false;
            for i in 0..n {
                let grid_knn = grid.find_k_nearest_with_scratch(&points, points[i], i, k, &mut scratch);
                let kiddo_knn = kiddo_find_k_nearest(&tree, &entries, points[i], i, k);

                let grid_set: HashSet<_> = grid_knn.iter().copied().collect();
                let kiddo_set: HashSet<_> = kiddo_knn.iter().copied().collect();

                let overlap = grid_set.intersection(&kiddo_set).count();
                worst_overlap = worst_overlap.min(overlap);

                if grid_set == kiddo_set {
                    exact_match += 1;
                } else {
                    // Check if grid is missing valid neighbors
                    let missing: Vec<_> = kiddo_set.difference(&grid_set).copied().collect();
                    let extra: Vec<_> = grid_set.difference(&kiddo_set).copied().collect();

                    if !missing.is_empty() {
                        missing_neighbors += 1;
                        // Verify the missing ones are actually closer
                        let query = points[i];
                        for &m in &missing {
                            let missing_dist = (points[m] - query).length_squared();
                            // Check if any extra neighbor is farther
                            for &e in &extra {
                                let extra_dist = (points[e] - query).length_squared();
                                if extra_dist > missing_dist + 1e-9 {
                                    wrong_neighbors += 1;
                                }
                            }
                        }

                        // Print first failure for debugging
                        if !first_failure_printed {
                            first_failure_printed = true;
                            let cell = grid.point_to_cell(query);
                            let neighbors = grid.cell_neighbors(cell);
                            let valid_neighbors: Vec<_> = neighbors.iter()
                                .filter(|&&c| c != u32::MAX)
                                .collect();
                            let total_candidates: usize = valid_neighbors.iter()
                                .map(|&&c| grid.cell_points(c as usize).len())
                                .sum();
                            println!("  First mismatch at i={}: cell={} neighbors={:?} candidates={}",
                                i, cell, valid_neighbors.len(), total_candidates);
                            println!("    grid_knn.len()={} grid_set.len()={} (DUPLICATES={})",
                                grid_knn.len(), grid_set.len(), grid_knn.len() - grid_set.len());
                            println!("    overlap={}, missing={}, extra={}",
                                overlap, missing.len(), extra.len());
                            // Print the neighbor cells to check for duplicates
                            println!("    neighbor_cells: {:?}", neighbors);
                        }
                    }
                }
            }

            let match_pct = exact_match as f64 / n as f64 * 100.0;
            println!("n={:>5} res={:>2}: exact={}/{} ({:.1}%) missing={} wrong={} worst_overlap={}/{}",
                n, res, exact_match, n, match_pct, missing_neighbors, wrong_neighbors, worst_overlap, k);

            // Strict correctness requirement
            assert_eq!(wrong_neighbors, 0, "Grid returned farther neighbors than kiddo!");
        }

        println!("\nAll tests passed - no incorrect neighbors returned.");
    }


    #[test]
    #[ignore] // Run with: cargo test test_cube_grid_vs_kiddo -- --ignored --nocapture
    fn test_cube_grid_vs_kiddo() {
        use std::time::Instant;

        println!("\n=== CubeMapGrid vs Kiddo k-NN Comparison ===\n");

        for &n in &[10_000usize, 100_000, 500_000, 1_000_000, 2_000_000] {
            let points = generate_test_points(n, 12345);
            let k = 24;

            // Target ~50 points per cell
            let res = ((n as f64 / 300.0).sqrt() as usize).max(4);

            // Build cube grid
            let t0 = Instant::now();
            let grid = CubeMapGrid::new(&points, res);
            let grid_build = t0.elapsed().as_secs_f64() * 1000.0;

            // Build kiddo tree
            let t0 = Instant::now();
            let (tree, entries) = build_kdtree(&points);
            let kiddo_build = t0.elapsed().as_secs_f64() * 1000.0;

            // Query timing - sample 1000 points
            let sample_size = 1000.min(n);

            let t0 = Instant::now();
            let mut scratch = grid.make_scratch();
            let mut grid_results: Vec<Vec<usize>> = Vec::with_capacity(sample_size);
            for i in 0..sample_size {
                grid_results.push(grid.find_k_nearest_with_scratch(&points, points[i], i, k, &mut scratch));
            }
            let grid_query = t0.elapsed().as_secs_f64() * 1000.0;

            let t0 = Instant::now();
            let kiddo_results: Vec<Vec<usize>> = (0..sample_size)
                .map(|i| kiddo_find_k_nearest(&tree, &entries, points[i], i, k))
                .collect();
            let kiddo_query = t0.elapsed().as_secs_f64() * 1000.0;

            // Compare results
            let mut exact_matches = 0;
            let mut partial_matches = 0;
            let mut total_overlap = 0usize;

            for (grid_knn, kiddo_knn) in grid_results.iter().zip(kiddo_results.iter()) {
                let grid_set: std::collections::HashSet<_> = grid_knn.iter().collect();
                let kiddo_set: std::collections::HashSet<_> = kiddo_knn.iter().collect();
                let overlap = grid_set.intersection(&kiddo_set).count();

                total_overlap += overlap;
                if overlap == k {
                    exact_matches += 1;
                } else if overlap >= k - 2 {
                    partial_matches += 1;
                }
            }

            let stats = grid.stats();

            println!("n={:>7} res={:>3} ({:.1} pts/cell, {} empty)",
                n, res, stats.avg_points_per_cell, stats.empty_cells);
            println!("  Build:  Grid={:>8.1}ms  Kiddo={:>8.1}ms  ({:.1}x)",
                grid_build, kiddo_build, kiddo_build / grid_build);
            println!("  Query:  Grid={:>8.1}ms  Kiddo={:>8.1}ms  ({:.1}x) [{} samples]",
                grid_query, kiddo_query, kiddo_query / grid_query, sample_size);
            // Extrapolated total time for all n queries
            let grid_total_est = grid_build + (grid_query / sample_size as f64) * n as f64;
            let kiddo_total_est = kiddo_build + (kiddo_query / sample_size as f64) * n as f64;

            println!("  Match:  exact={}/{} partial={} avg_overlap={:.1}/{}",
                exact_matches, sample_size, partial_matches,
                total_overlap as f64 / sample_size as f64, k);
            println!("  Est total (build + all queries): Grid={:.0}ms  Kiddo={:.0}ms",
                grid_total_est, kiddo_total_est);
            println!();
        }
    }

    #[test]
    #[ignore] // Run with: cargo test test_cube_grid_parallel -- --ignored --nocapture
    fn test_cube_grid_parallel() {
        use rayon::prelude::*;
        use std::time::Instant;
        use std::sync::atomic::{AtomicUsize, Ordering};

        println!("\n=== Parallel k-NN: CubeMapGrid vs Kiddo ===\n");

        // Test one size at a time to reduce memory pressure
        for &n in &[100_000usize, 500_000, 1_000_000, 2_000_000] {
            let points = generate_test_points(n, 12345);
            let k = 24;
            let res = ((n as f64 / 300.0).sqrt() as usize).max(4);

            // Grid: build + query (don't store results, just count)
            let t0 = Instant::now();
            let grid = CubeMapGrid::new(&points, res);
            let grid_build = t0.elapsed().as_secs_f64() * 1000.0;

            let count = AtomicUsize::new(0);
            let t0 = Instant::now();
            (0..n)
                .into_par_iter()
                .for_each_init(|| grid.make_scratch(), |scratch, i| {
                    let knn = grid.find_k_nearest_with_scratch(&points, points[i], i, k, scratch);
                    count.fetch_add(knn.len(), Ordering::Relaxed);
                });
            let grid_query = t0.elapsed().as_secs_f64() * 1000.0;
            let _ = count.load(Ordering::Relaxed); // prevent optimization

            drop(grid); // Free memory before kiddo

            // Kiddo: build + query
            let t0 = Instant::now();
            let (tree, entries) = build_kdtree(&points);
            let kiddo_build = t0.elapsed().as_secs_f64() * 1000.0;

            let count = AtomicUsize::new(0);
            let t0 = Instant::now();
            (0..n).into_par_iter().for_each(|i| {
                let knn = kiddo_find_k_nearest(&tree, &entries, points[i], i, k);
                count.fetch_add(knn.len(), Ordering::Relaxed);
            });
            let kiddo_query = t0.elapsed().as_secs_f64() * 1000.0;
            let _ = count.load(Ordering::Relaxed);

            let grid_total = grid_build + grid_query;
            let kiddo_total = kiddo_build + kiddo_query;

            println!("n={:>7}:", n);
            println!("  Grid:  build={:>7.1}ms  query={:>7.1}ms  total={:>7.1}ms",
                grid_build, grid_query, grid_total);
            println!("  Kiddo: build={:>7.1}ms  query={:>7.1}ms  total={:>7.1}ms",
                kiddo_build, kiddo_query, kiddo_total);
            println!("  Speedup: {:.2}x", kiddo_total / grid_total);
            println!();
        }
    }
}
