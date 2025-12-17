use glam::Vec3;
use std::collections::HashMap;

use super::ConvexHull;

/// A view into a single Voronoi cell, providing access to its data.
#[derive(Debug, Clone, Copy)]
pub struct CellView<'a> {
    /// Index of the generator point for this cell.
    pub generator_index: usize,
    /// Indices into the `vertices` array of SphericalVoronoi.
    /// Ordered counter-clockwise when viewed from outside the sphere.
    pub vertex_indices: &'a [usize],
}

impl<'a> CellView<'a> {
    /// Number of vertices in this cell.
    #[inline]
    pub fn len(&self) -> usize {
        self.vertex_indices.len()
    }

    /// Whether the cell has no vertices.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertex_indices.is_empty()
    }
}

/// A single Voronoi cell on the sphere (internal storage).
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    /// Index of the generator point for this cell.
    pub generator_index: usize,
    /// Start index into the flat `cell_indices` buffer.
    vertex_start: usize,
    /// Number of vertices for this cell.
    vertex_count: usize,
}

impl VoronoiCell {
    /// Create a new VoronoiCell with the given parameters.
    #[inline]
    pub fn new(generator_index: usize, vertex_start: usize, vertex_count: usize) -> Self {
        Self {
            generator_index,
            vertex_start,
            vertex_count,
        }
    }
}

/// A spherical Voronoi diagram.
#[derive(Debug)]
pub struct SphericalVoronoi {
    /// The generator points (input points on the sphere).
    pub generators: Vec<Vec3>,
    /// The Voronoi vertices (circumcenters of hull facets, projected to sphere).
    pub vertices: Vec<Vec3>,
    /// The Voronoi cells, one per generator.
    cells: Vec<VoronoiCell>,
    /// Flat buffer of all cell vertex indices.
    cell_indices: Vec<usize>,
}

impl SphericalVoronoi {
    /// Create a SphericalVoronoi directly from pre-built components.
    ///
    /// This is the most efficient constructor when you've already built the
    /// cells and cell_indices buffers directly.
    #[inline]
    pub fn from_raw_parts(
        generators: Vec<Vec3>,
        vertices: Vec<Vec3>,
        cells: Vec<VoronoiCell>,
        cell_indices: Vec<usize>,
    ) -> Self {
        Self {
            generators,
            vertices,
            cells,
            cell_indices,
        }
    }

    /// Create a new SphericalVoronoi from components.
    pub fn new(
        generators: Vec<Vec3>,
        vertices: Vec<Vec3>,
        cell_data: Vec<(usize, Vec<usize>)>, // (generator_index, vertex_indices) per cell
    ) -> Self {
        let mut cells = Vec::with_capacity(cell_data.len());
        let total_indices: usize = cell_data.iter().map(|(_, v)| v.len()).sum();
        let mut cell_indices = Vec::with_capacity(total_indices);

        for (generator_index, vertex_vec) in cell_data {
            let vertex_start = cell_indices.len();
            let vertex_count = vertex_vec.len();
            cell_indices.extend(vertex_vec);
            cells.push(VoronoiCell {
                generator_index,
                vertex_start,
                vertex_count,
            });
        }

        SphericalVoronoi {
            generators,
            vertices,
            cells,
            cell_indices,
        }
    }

    /// Create a new SphericalVoronoi with a builder function that populates cells directly.
    ///
    /// The builder receives a mutable reference to the internal state and should call
    /// `add_cell` for each cell to build directly into the flat buffer.
    /// Returns (voronoi, vertices) where vertices is the Vec built by the builder.
    pub fn build_direct<F>(generators: Vec<Vec3>, num_cells: usize, builder: F) -> Self
    where
        F: FnOnce(&mut SphericalVoronoiBuilder),
    {
        let mut b = SphericalVoronoiBuilder {
            cells: Vec::with_capacity(num_cells),
            cell_indices: Vec::with_capacity(num_cells * 6), // ~6 vertices per cell on average
            vertices: Vec::with_capacity(num_cells * 2),     // ~2 unique vertices per cell
        };
        builder(&mut b);
        SphericalVoronoi {
            generators,
            vertices: b.vertices,
            cells: b.cells,
            cell_indices: b.cell_indices,
        }
    }
}

/// Builder for constructing SphericalVoronoi directly without intermediate allocations.
pub struct SphericalVoronoiBuilder {
    cells: Vec<VoronoiCell>,
    cell_indices: Vec<usize>,
    /// Vertices collected during construction.
    pub vertices: Vec<Vec3>,
}

impl SphericalVoronoiBuilder {
    /// Add a cell with its vertex indices directly to the flat buffer.
    #[inline]
    pub fn add_cell<I: IntoIterator<Item = usize>>(
        &mut self,
        generator_index: usize,
        vertex_indices: I,
    ) {
        let vertex_start = self.cell_indices.len();
        self.cell_indices.extend(vertex_indices);
        let vertex_count = self.cell_indices.len() - vertex_start;
        self.cells.push(VoronoiCell {
            generator_index,
            vertex_start,
            vertex_count,
        });
    }

    /// Add a vertex and return its index.
    #[inline]
    pub fn add_vertex(&mut self, pos: Vec3) -> usize {
        let idx = self.vertices.len();
        self.vertices.push(pos);
        idx
    }
}

impl SphericalVoronoi {
    /// Number of cells in the diagram.
    #[inline]
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// Get a view of cell at the given index.
    #[inline]
    pub fn cell(&self, idx: usize) -> CellView<'_> {
        let cell = &self.cells[idx];
        CellView {
            generator_index: cell.generator_index,
            vertex_indices: &self.cell_indices
                [cell.vertex_start..cell.vertex_start + cell.vertex_count],
        }
    }

    /// Iterate over all cells as views.
    #[inline]
    pub fn iter_cells(&self) -> impl Iterator<Item = CellView<'_>> {
        self.cells.iter().map(move |cell| CellView {
            generator_index: cell.generator_index,
            vertex_indices: &self.cell_indices
                [cell.vertex_start..cell.vertex_start + cell.vertex_count],
        })
    }

    /// Compute the spherical Voronoi diagram from points on a unit sphere.
    ///
    /// The key insight: for points on a sphere centered at origin,
    /// the convex hull facets form the Delaunay triangulation,
    /// and the dual graph is the Voronoi diagram.
    pub fn compute(points: &[Vec3]) -> Self {
        let hull = ConvexHull::compute(points);

        // Step 1: Compute Voronoi vertices (circumcenters of hull facets)
        let vertices: Vec<Vec3> = hull
            .facets
            .iter()
            .map(|facet| {
                let a = points[facet.indices[0]];
                let b = points[facet.indices[1]];
                let c = points[facet.indices[2]];
                circumcenter_on_sphere(a, b, c)
            })
            .collect();

        // Step 2: Build adjacency: for each point, find all facets containing it
        let mut point_to_facets: HashMap<usize, Vec<usize>> = HashMap::new();
        for (facet_idx, facet) in hull.facets.iter().enumerate() {
            for &point_idx in &facet.indices {
                point_to_facets
                    .entry(point_idx)
                    .or_default()
                    .push(facet_idx);
            }
        }

        // Step 3: Build cells by ordering vertices CCW around each generator (parallel)
        use rayon::prelude::*;
        let cell_data: Vec<(usize, Vec<usize>)> = (0..points.len())
            .into_par_iter()
            .map(|point_idx| {
                let facet_indices = point_to_facets.get(&point_idx).cloned().unwrap_or_default();
                let ordered = order_vertices_ccw(points[point_idx], &facet_indices, &vertices);
                (point_idx, ordered)
            })
            .collect();

        Self::new(points.to_vec(), vertices, cell_data)
    }
}

/// Compute the circumcenter of a spherical triangle and project it to the sphere.
fn circumcenter_on_sphere(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    // The circumcenter is perpendicular to both (b-a) and (c-a)
    // i.e., it's in the direction of (b-a) × (c-a)
    let ab = b - a;
    let ac = c - a;
    let normal = ab.cross(ac);

    // Normalize to project to sphere
    // The sign determines which hemisphere; we want the one "outside" the hull
    // For convex hull of sphere points, the facet normal points outward
    let center = normal.normalize();

    // Check if the center is on the correct side (same hemisphere as the triangle centroid)
    let centroid = (a + b + c).normalize();
    if center.dot(centroid) < 0.0 {
        -center
    } else {
        center
    }
}

/// Order vertex indices counter-clockwise around a generator point when viewed from outside.
fn order_vertices_ccw(generator: Vec3, vertex_indices: &[usize], vertices: &[Vec3]) -> Vec<usize> {
    if vertex_indices.len() <= 2 {
        return vertex_indices.to_vec();
    }

    // Project vertices to tangent plane at generator and sort by angle
    let mut indexed: Vec<(usize, f32)> = vertex_indices
        .iter()
        .map(|&idx| {
            let v = vertices[idx];
            let angle = angle_in_tangent_plane(generator, v);
            (idx, angle)
        })
        .collect();

    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    indexed.into_iter().map(|(idx, _)| idx).collect()
}

/// Compute the angle of a point in the tangent plane at the generator.
fn angle_in_tangent_plane(generator: Vec3, point: Vec3) -> f32 {
    // Create local coordinate system on tangent plane
    // Use an arbitrary reference direction perpendicular to generator
    let up = if generator.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };

    let tangent_x = generator.cross(up).normalize();
    let tangent_y = generator.cross(tangent_x).normalize();

    // Project point onto tangent plane
    let to_point = point - generator * generator.dot(point);

    let x = to_point.dot(tangent_x);
    let y = to_point.dot(tangent_y);

    y.atan2(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::random_sphere_points;

    #[test]
    fn test_voronoi_basic() {
        let points = random_sphere_points(20);
        let voronoi = SphericalVoronoi::compute(&points);

        assert_eq!(voronoi.num_cells(), 20);
        assert_eq!(voronoi.generators.len(), 20);

        // Each cell should have at least 3 vertices
        for cell in voronoi.iter_cells() {
            assert!(
                cell.vertex_indices.len() >= 3,
                "Cell has {} vertices",
                cell.vertex_indices.len()
            );
        }
    }
}
