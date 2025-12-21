//! Cube-map based spatial grid for fast spatial queries on unit sphere.
//!
//! Projects sphere onto 6 cube faces, divides each into a regular grid.
//! O(n) build, O(1) cell lookup.
//!
//! Supports two query types:
//! - `knn_into`: k-nearest neighbors
//! - `within_cos_into`: all points within angular distance (range query)
//!
//! Queries use best-first expansion over neighboring cells with conservative
//! distance bounds. Typical uniform inputs terminate after visiting a handful
//! of cells; worst-case falls back to brute force.

#[cfg(test)]
mod tests;

use glam::Vec3;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Lazily yields neighbors in sorted order (closest first).
///
/// Construction: O(n) via heapify
/// Each next(): O(log n) heap pop
///
/// This is the "robust" fallback - guaranteed correct, no cell heuristics.
pub struct LazyNeighborIter {
    heap: BinaryHeap<Reverse<(OrdF32, u32)>>,
}

impl Iterator for LazyNeighborIter {
    type Item = (usize, f32); // (index, distance_squared)

    fn next(&mut self) -> Option<Self::Item> {
        self.heap.pop().map(|Reverse((d, i))| (i as usize, d.get()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.heap.len(), Some(self.heap.len()))
    }
}

impl ExactSizeIterator for LazyNeighborIter {}

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
    pub(super) res: usize,
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
    pub(super) cell_centers: Vec<Vec3>,
    /// Spherical cap radius around `cell_centers[cell]` that conservatively contains the cell.
    /// Stored as cos/sin for fast per-query bounds.
    pub(super) cell_cos_radius: Vec<f32>,
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
pub(super) fn face_uv_to_3d(face: usize, u: f32, v: f32) -> Vec3 {
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
pub(super) fn cell_to_face_ij(cell: usize, res: usize) -> (usize, usize, usize) {
    let face = cell / (res * res);
    let rem = cell % (res * res);
    let iv = rem / res;
    let iu = rem % res;
    (face, iu, iv)
}

/// Reusable per-query scratch buffers.
///
/// For performance (especially parallel queries), prefer `CubeMapGrid::make_scratch()`.
pub struct CubeMapGridScratch {
    /// Cell visitation stamps (avoids clearing between queries)
    visited_stamp: Vec<u32>,
    stamp: u32,
    /// Priority queue for cell expansion (min-heap by distance bound)
    cell_heap: BinaryHeap<Reverse<(OrdF32, u32)>>,
    /// Candidate heap for k-NN (max-heap: furthest at top for easy eviction)
    /// (dist_sq, point_idx)
    knn_heap: BinaryHeap<(OrdF32, u32)>,
    /// Current k limit for knn_heap
    knn_k: usize,
}

impl CubeMapGridScratch {
    pub fn new(num_cells: usize) -> Self {
        Self {
            visited_stamp: vec![0; num_cells],
            stamp: 0,
            cell_heap: BinaryHeap::new(),
            knn_heap: BinaryHeap::new(),
            knn_k: 0,
        }
    }

    #[inline]
    fn begin_query(&mut self) {
        self.cell_heap.clear();
        self.knn_heap.clear();
        self.knn_k = 0;

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
    fn pop_cell(&mut self) -> Option<(f32, u32)> {
        self.cell_heap.pop().map(|Reverse((bound, cell))| (bound.get(), cell))
    }

    /// Get the distance to the k-th candidate (for pruning).
    /// Returns infinity if we have fewer than k candidates.
    #[inline]
    fn kth_dist_sq(&self) -> f32 {
        if self.knn_heap.len() < self.knn_k {
            f32::INFINITY
        } else {
            self.knn_heap.peek().map(|(d, _)| d.get()).unwrap_or(f32::INFINITY)
        }
    }

    /// Try to add a neighbor to the k-NN heap.
    #[inline]
    fn try_add_knn(&mut self, idx: usize, dist_sq: f32) {
        if dist_sq.is_nan() {
            return;
        }
        let idx_u32 = idx as u32;

        if self.knn_heap.len() < self.knn_k {
            // Haven't found k neighbors yet - always add
            self.knn_heap.push((OrdF32::new(dist_sq), idx_u32));
        } else if let Some(&(top_dist, _)) = self.knn_heap.peek() {
            // Only add if closer than the furthest current neighbor
            if dist_sq < top_dist.get() {
                self.knn_heap.pop();
                self.knn_heap.push((OrdF32::new(dist_sq), idx_u32));
            }
        }
    }

    /// Copy k-NN results into output vec (sorted by distance, closest first).
    fn copy_knn_into(&mut self, out: &mut Vec<usize>) {
        out.clear();
        out.reserve(self.knn_heap.len());

        // Drain heap into vec (comes out in reverse order - furthest first)
        let mut temp: Vec<(OrdF32, u32)> = self.knn_heap.drain().collect();
        // Sort by distance ascending
        temp.sort_by(|a, b| a.0.cmp(&b.0));

        for (_, idx) in temp {
            out.push(idx as usize);
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
        let k = k.min(n - 1);

        let num_cells = 6 * self.res * self.res;
        scratch.begin_query();
        scratch.knn_k = k;

        let start_cell = self.point_to_cell(query) as u32;
        scratch.mark_visited(start_cell);
        scratch.push_cell(start_cell, 0.0);

        // If the search balloons, brute force is usually cheaper than touching most cells
        let max_cells_before_bruteforce = (num_cells / 2).max(64);
        let mut visited_cells = 0usize;

        while let Some((bound_dist_sq, cell_u32)) = scratch.pop_cell() {
            // Prune: if next cell's bound >= k-th best distance, we're done
            let kth_dist = scratch.kth_dist_sq();
            if scratch.knn_heap.len() >= k && bound_dist_sq >= kth_dist {
                break;
            }

            visited_cells += 1;
            if visited_cells > max_cells_before_bruteforce {
                self.bruteforce_knn(points, query, query_idx, k, scratch);
                scratch.copy_knn_into(out_indices);
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
                scratch.try_add_knn(pidx, dist_sq);
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

        if scratch.knn_heap.len() < k {
            self.bruteforce_knn(points, query, query_idx, k, scratch);
        }

        scratch.copy_knn_into(out_indices);
    }

    /// Range query: find all points within angular distance of query.
    ///
    /// `min_cos` is the cosine threshold: points with `query.dot(point) >= min_cos` are included.
    /// For angular distance θ, use `min_cos = cos(θ)`.
    ///
    /// Results are NOT sorted by distance (for efficiency).
    pub fn within_cos_into(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        min_cos: f32,
        scratch: &mut CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) {
        let n = points.len();
        out_indices.clear();

        if n <= 1 {
            return;
        }

        // Convert min_cos to max_dist_sq on unit sphere
        // For unit vectors: |a - b|² = 2 - 2*a·b = 2*(1 - cos_angle)
        let max_dist_sq = 2.0 * (1.0 - min_cos);

        let num_cells = 6 * self.res * self.res;
        scratch.begin_query();

        let start_cell = self.point_to_cell(query) as u32;
        scratch.mark_visited(start_cell);
        scratch.push_cell(start_cell, 0.0);

        // If the search balloons, brute force scan
        let max_cells_before_bruteforce = (num_cells / 2).max(64);
        let mut visited_cells = 0usize;

        while let Some((bound_dist_sq, cell_u32)) = scratch.pop_cell() {
            // Prune: if next cell's min bound exceeds threshold, we're done
            if bound_dist_sq > max_dist_sq {
                break;
            }

            visited_cells += 1;
            if visited_cells > max_cells_before_bruteforce {
                // Fall back to brute force
                out_indices.clear();
                for (idx, p) in points.iter().enumerate() {
                    if idx == query_idx {
                        continue;
                    }
                    if query.dot(*p) >= min_cos {
                        out_indices.push(idx);
                    }
                }
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
                let cos_angle = query.dot(points[pidx]);
                if cos_angle >= min_cos {
                    out_indices.push(pidx);
                }
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

    /// Returns a lazy iterator over ALL neighbors sorted by distance.
    ///
    /// O(n) construction via heapify, O(log n) per next() call.
    /// Ignores the grid structure entirely - guaranteed correct fallback.
    ///
    /// Each item is `(index, distance_squared)`.
    pub fn iter_neighbors_sorted(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
    ) -> LazyNeighborIter {
        let items: Vec<_> = points
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != query_idx)
            .map(|(i, p)| Reverse((OrdF32::new((*p - query).length_squared()), i as u32)))
            .collect();

        LazyNeighborIter {
            heap: BinaryHeap::from(items), // O(n) heapify
        }
    }

    /// Brute-force scan for k-NN when grid expansion is too expensive.
    fn bruteforce_knn(
        &self,
        points: &[Vec3],
        query: Vec3,
        query_idx: usize,
        k: usize,
        scratch: &mut CubeMapGridScratch,
    ) {
        scratch.knn_heap.clear();
        scratch.knn_k = k;
        for (idx, p) in points.iter().enumerate() {
            if idx == query_idx {
                continue;
            }
            let dist_sq = (*p - query).length_squared();
            scratch.try_add_knn(idx, dist_sq);
        }
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
