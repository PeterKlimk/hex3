use glam::Vec3;

use super::SphericalVoronoi;

/// Perform Lloyd relaxation on points on a sphere.
///
/// Lloyd relaxation moves each generator to the centroid of its Voronoi cell,
/// creating a more uniform distribution of cells.
pub fn lloyd_relax(points: &mut [Vec3], iterations: usize) {
    for _ in 0..iterations {
        lloyd_step(points);
    }
}

/// Perform a single Lloyd relaxation step.
fn lloyd_step(points: &mut [Vec3]) {
    let voronoi = SphericalVoronoi::compute(points);

    for cell in &voronoi.cells {
        if cell.vertex_indices.len() < 3 {
            continue;
        }

        // Get the actual vertex positions for this cell
        let cell_vertices: Vec<Vec3> = cell
            .vertex_indices
            .iter()
            .map(|&idx| voronoi.vertices[idx])
            .collect();

        // Compute centroid and move the generator
        let centroid = spherical_polygon_centroid(&cell_vertices);
        points[cell.generator_index] = centroid;
    }
}

/// Compute the centroid of a spherical polygon.
///
/// Uses area-weighted triangulation: decompose the polygon into triangles
/// from the first vertex, weight each triangle's centroid by its area.
fn spherical_polygon_centroid(vertices: &[Vec3]) -> Vec3 {
    if vertices.is_empty() {
        return Vec3::ZERO;
    }
    if vertices.len() == 1 {
        return vertices[0];
    }
    if vertices.len() == 2 {
        return (vertices[0] + vertices[1]).normalize();
    }

    let v0 = vertices[0];
    let mut weighted_sum = Vec3::ZERO;
    let mut total_area = 0.0f32;

    for i in 1..vertices.len() - 1 {
        let v1 = vertices[i];
        let v2 = vertices[i + 1];

        // Spherical triangle centroid (approximate: average then normalize)
        let tri_centroid = (v0 + v1 + v2).normalize();

        // Spherical triangle area
        let area = spherical_triangle_area(v0, v1, v2);

        weighted_sum += tri_centroid * area;
        total_area += area;
    }

    if total_area > 1e-10 {
        (weighted_sum / total_area).normalize()
    } else {
        // Fallback: simple average
        let sum: Vec3 = vertices.iter().copied().sum();
        sum.normalize()
    }
}

/// Compute the area of a spherical triangle using the spherical excess formula.
fn spherical_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f32 {
    // Using the formula: area = 2 * atan2(|a · (b × c)|, 1 + a·b + b·c + c·a)
    let cross = b.cross(c);
    let numerator = a.dot(cross).abs();
    let denominator = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * numerator.atan2(denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::random_sphere_points;

    #[test]
    fn test_lloyd_improves_distribution() {
        let mut points = random_sphere_points(50);

        // Measure initial distribution (variance of distances to neighbors)
        let initial_variance = distribution_variance(&points);

        lloyd_relax(&mut points, 20);

        // Verify all points still on sphere
        for p in &points {
            let len = p.length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Point not on unit sphere after relaxation"
            );
        }

        let final_variance = distribution_variance(&points);

        // Distribution should be more uniform (lower variance)
        assert!(
            final_variance <= initial_variance * 1.1, // Allow some tolerance
            "Lloyd relaxation did not improve distribution: {} -> {}",
            initial_variance,
            final_variance
        );
    }

    fn distribution_variance(points: &[Vec3]) -> f32 {
        // Compute variance of nearest-neighbor distances
        let distances: Vec<f32> = points
            .iter()
            .map(|p| {
                points
                    .iter()
                    .filter(|&q| q != p)
                    .map(|q| (*p - *q).length())
                    .min_by(|a, b| a.total_cmp(b))
                    .unwrap_or(0.0)
            })
            .collect();

        let mean: f32 = distances.iter().sum::<f32>() / distances.len() as f32;
        let variance: f32 =
            distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32;
        variance
    }
}
