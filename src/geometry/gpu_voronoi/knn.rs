//! K-nearest neighbor provider for Voronoi cell construction.

use glam::Vec3;

pub use crate::geometry::cube_grid::{IterScratch, KnnQuery};

/// K-NN provider using CubeMapGrid - O(n) build time, good for large point sets.
///
/// Wraps a CubeMapGrid and provides confidence-based k-NN queries with const-generic
/// fixed buffers for zero per-query allocation.
pub struct CubeMapGridKnn<'a> {
    grid: crate::geometry::cube_grid::CubeMapGrid,
    points: &'a [Vec3],
}

impl<'a> CubeMapGridKnn<'a> {
    pub fn new(points: &'a [Vec3]) -> Self {
        let n = points.len();
        const TARGET_POINTS_PER_CELL: f64 = 8.0;
        let res = ((n as f64 / (6.0 * TARGET_POINTS_PER_CELL)).sqrt() as usize).max(4);
        let grid = crate::geometry::cube_grid::CubeMapGrid::new(points, res);
        Self { grid, points }
    }

    /// Create a scratch buffer for k-NN queries.
    #[inline]
    pub fn make_iter_scratch(&self) -> IterScratch {
        self.grid.make_iter_scratch()
    }

    /// Create a k-NN query with confidence-based incremental fetching.
    ///
    /// Use `fetch()` to get batches of neighbors as they become confident.
    /// Each batch is sorted by distance (closest first).
    ///
    /// Returns `(dot_product, neighbor_index)` pairs where higher dot = closer.
    ///
    /// # Example
    /// ```ignore
    /// let mut scratch = knn.make_iter_scratch();
    /// let mut query = knn.knn_query::<48>(pt, idx, &mut scratch);
    ///
    /// 'outer: while let Some(batch) = query.fetch() {
    ///     for &(dot, neighbor_idx) in batch {
    ///         builder.clip(neighbor_idx as usize, points[neighbor_idx as usize]);
    ///         if builder.can_terminate(dot) {
    ///             break 'outer;
    ///         }
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn knn_query<'s, const MAX_K: usize>(
        &'a self,
        query: Vec3,
        query_idx: usize,
        scratch: &'s mut IterScratch,
    ) -> KnnQuery<'a, 's, MAX_K> {
        self.grid
            .knn_query::<MAX_K>(self.points, query, query_idx, scratch)
    }

    /// Access the underlying points slice.
    #[inline]
    pub fn points(&self) -> &[Vec3] {
        self.points
    }

    // =========================================================================
    // Convenience methods for backward compatibility (used by validation code)
    // =========================================================================

    /// Create a legacy scratch buffer (for backward compatibility with old API).
    #[inline]
    pub fn make_scratch(&self) -> crate::geometry::cube_grid::CubeMapGridScratch {
        self.grid.make_scratch()
    }

    /// Find the k nearest neighbors to the query point (backward compatibility).
    ///
    /// This uses the old non-incremental API. For new code, prefer `knn_query()`.
    #[inline]
    pub fn knn_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        scratch: &mut crate::geometry::cube_grid::CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) {
        out_indices.clear();
        if k == 0 {
            return;
        }
        self.grid.find_k_nearest_with_scratch_into_dot_topk(
            self.points,
            query,
            query_idx,
            k,
            scratch,
            out_indices,
        );
    }
}
