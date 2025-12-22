//! K-nearest neighbor providers for Voronoi cell construction.

use glam::Vec3;

pub use crate::geometry::cube_grid::KnnStatus;

/// Trait for k-nearest neighbor queries.
pub trait KnnProvider: Sync {
    type Scratch: Send;

    fn make_scratch(&self) -> Self::Scratch;

    /// Find the k nearest neighbors to the query point.
    fn knn_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        scratch: &mut Self::Scratch,
        out_indices: &mut Vec<usize>,
    );

    /// Start a resumable k-NN query.
    ///
    /// `track_limit` controls how many candidates are tracked internally.
    /// Can resume up to k=track_limit; beyond that, a fresh query is needed.
    fn knn_resumable_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut Self::Scratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus;

    /// Resume a k-NN query to fetch additional neighbors.
    ///
    /// `new_k` should be larger than the previous k but within the original `track_limit`.
    fn knn_resume_into(
        &self,
        query: Vec3,
        query_idx: usize,
        new_k: usize,
        scratch: &mut Self::Scratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus;
}

/// K-NN provider using CubeMapGrid - O(n) build time, good for large point sets.
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
}

impl<'a> KnnProvider for CubeMapGridKnn<'a> {
    type Scratch = crate::geometry::cube_grid::CubeMapGridScratch;

    #[inline]
    fn make_scratch(&self) -> Self::Scratch {
        self.grid.make_scratch()
    }

    #[inline]
    fn knn_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        scratch: &mut Self::Scratch,
        out_indices: &mut Vec<usize>,
    ) {
        if k == 0 {
            out_indices.clear();
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

    #[inline]
    fn knn_resumable_into(
        &self,
        query: Vec3,
        query_idx: usize,
        k: usize,
        track_limit: usize,
        scratch: &mut Self::Scratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        if k == 0 {
            out_indices.clear();
            return KnnStatus::Exhausted;
        }
        self.grid.find_k_nearest_resumable_into(
            self.points,
            query,
            query_idx,
            k,
            track_limit,
            scratch,
            out_indices,
        )
    }

    #[inline]
    fn knn_resume_into(
        &self,
        query: Vec3,
        query_idx: usize,
        new_k: usize,
        scratch: &mut Self::Scratch,
        out_indices: &mut Vec<usize>,
    ) -> KnnStatus {
        if new_k == 0 {
            out_indices.clear();
            return KnnStatus::Exhausted;
        }
        self.grid
            .resume_k_nearest_into(self.points, query, query_idx, new_k, scratch, out_indices)
    }
}
