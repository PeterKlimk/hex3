use glam::Vec3;
use std::collections::HashMap;

use super::ConvexHull;

/// A single Voronoi cell on the sphere.
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    /// Index of the generator point for this cell.
    pub generator_index: usize,
    /// Indices into the `vertices` array of SphericalVoronoi.
    /// Ordered counter-clockwise when viewed from outside the sphere.
    pub vertex_indices: Vec<usize>,
}

/// A spherical Voronoi diagram.
#[derive(Debug)]
pub struct SphericalVoronoi {
    /// The generator points (input points on the sphere).
    pub generators: Vec<Vec3>,
    /// The Voronoi vertices (circumcenters of hull facets, projected to sphere).
    pub vertices: Vec<Vec3>,
    /// The Voronoi cells, one per generator.
    pub cells: Vec<VoronoiCell>,
}

impl SphericalVoronoi {
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

        // Step 3: Build cells by ordering vertices CCW around each generator
        let cells: Vec<VoronoiCell> = (0..points.len())
            .map(|point_idx| {
                let facet_indices = point_to_facets.get(&point_idx).cloned().unwrap_or_default();
                let cell_vertices: Vec<usize> = facet_indices;

                // Order vertices counter-clockwise around the generator
                let ordered = order_vertices_ccw(points[point_idx], &cell_vertices, &vertices);

                VoronoiCell {
                    generator_index: point_idx,
                    vertex_indices: ordered,
                }
            })
            .collect();

        SphericalVoronoi {
            generators: points.to_vec(),
            vertices,
            cells,
        }
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

        assert_eq!(voronoi.cells.len(), 20);
        assert_eq!(voronoi.generators.len(), 20);

        // Each cell should have at least 3 vertices
        for cell in &voronoi.cells {
            assert!(
                cell.vertex_indices.len() >= 3,
                "Cell has {} vertices",
                cell.vertex_indices.len()
            );
        }
    }
}
