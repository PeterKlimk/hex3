//! K-nearest neighbor provider for Voronoi cell construction.

use glam::{Vec3, Vec3A};

pub use crate::geometry::cube_grid::{IterScratch, KnnQuery};

/// K-NN provider using CubeMapGrid - O(n) build time, good for large point sets.
///
/// Wraps a CubeMapGrid and provides confidence-based k-NN queries with const-generic
/// fixed buffers for zero per-query allocation.
pub struct CubeMapGridKnn<'a> {
    grid: crate::geometry::cube_grid::CubeMapGrid,
    points: &'a [Vec3],
    points_a: Vec<Vec3A>,
}

impl<'a> CubeMapGridKnn<'a> {
    pub fn new(points: &'a [Vec3]) -> Self {
        let n = points.len();
        const TARGET_POINTS_PER_CELL: f64 = 16.0;
        let res = ((n as f64 / (6.0 * TARGET_POINTS_PER_CELL)).sqrt() as usize).max(4);
        let grid = crate::geometry::cube_grid::CubeMapGrid::new(points, res);
        let points_a: Vec<Vec3A> = points.iter().map(|&p| p.into()).collect();
        Self {
            grid,
            points,
            points_a,
        }
    }

    /// Create a scratch buffer for k-NN queries.
    #[inline]
    pub fn make_iter_scratch(&self) -> IterScratch {
        self.grid.make_iter_scratch()
    }

    /// Create a k-NN query with confidence-based incremental fetching.
    ///
    /// Deprecated for production use: prefer resumable queries for better performance
    /// and more stable early-termination behavior.
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

    /// Start a resumable k-NN query, tracking up to `track_limit` neighbors.
    #[inline]
    pub fn knn_resumable_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut crate::geometry::cube_grid::CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> crate::geometry::cube_grid::KnnStatus {
        self.grid.find_k_nearest_resumable_into_vec3a(
            &self.points_a,
            query.into(),
            query_idx,
            k,
            track_limit,
            scratch,
            out_indices,
        )
    }

    /// Resume a resumable k-NN query to fetch additional neighbors.
    #[inline]
    pub fn knn_resume_into(
        &self,
        query: Vec3,
        query_idx: usize,
        new_k: usize,
        scratch: &mut crate::geometry::cube_grid::CubeMapGridScratch,
        out_indices: &mut Vec<usize>,
    ) -> crate::geometry::cube_grid::KnnStatus {
        self.grid.resume_k_nearest_into_vec3a(
            &self.points_a,
            query.into(),
            query_idx,
            new_k,
            scratch,
            out_indices,
        )
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
        self.grid.find_k_nearest_with_scratch_into_dot_topk_vec3a(
            &self.points_a,
            query.into(),
            query_idx,
            k,
            scratch,
            out_indices,
        );
    }
}
