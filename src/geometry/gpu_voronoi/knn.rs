//! K-nearest neighbor providers for Voronoi cell construction.

use glam::Vec3;

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

    /// Find all neighbors within a cosine threshold of the query point.
    ///
    /// `min_cos` is the minimum dot product (cosine of angle) required.
    /// Points where `query.dot(point) >= min_cos` are included.
    fn within_cos_into(
        &self,
        query: Vec3,
        query_idx: usize,
        min_cos: f32,
        scratch: &mut Self::Scratch,
        out_indices: &mut Vec<usize>,
    );
}

/// K-NN provider using CubeMapGrid - O(n) build time, good for large point sets.
pub struct CubeMapGridKnn<'a> {
    grid: crate::geometry::cube_grid::CubeMapGrid,
    points: &'a [Vec3],
}

impl<'a> CubeMapGridKnn<'a> {
    pub fn new(points: &'a [Vec3]) -> Self {
        let n = points.len();
        let res = ((n as f64 / 300.0).sqrt() as usize).max(4);
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
        self.grid.find_k_nearest_with_scratch_into(
            self.points,
            query,
            query_idx,
            k,
            scratch,
            out_indices,
        );
    }

    #[inline]
    fn within_cos_into(
        &self,
        query: Vec3,
        query_idx: usize,
        min_cos: f32,
        scratch: &mut Self::Scratch,
        out_indices: &mut Vec<usize>,
    ) {
        self.grid.within_cos_into(
            self.points,
            query,
            query_idx,
            min_cos,
            scratch,
            out_indices,
        );
    }
}
