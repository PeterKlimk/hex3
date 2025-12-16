use glam::Vec3;
use qhull_enhanced::Qh;

/// A triangular facet of the convex hull, with indices into the original point array.
/// Winding is counter-clockwise when viewed from outside.
#[derive(Debug, Clone)]
pub struct HullFacet {
    pub indices: [usize; 3],
}

/// Result of computing a 3D convex hull.
#[derive(Debug)]
pub struct ConvexHull {
    pub facets: Vec<HullFacet>,
}

impl ConvexHull {
    /// Compute the convex hull of a set of 3D points.
    pub fn compute(points: &[Vec3]) -> Self {
        // qhull expects iterables of [f64; N]
        let pts: Vec<[f64; 3]> = points
            .iter()
            .map(|p| [p.x as f64, p.y as f64, p.z as f64])
            .collect();

        // Optimize for sphere points: all points are on the hull (no interior),
        // uniformly distributed (not narrow)
        let qh = Qh::builder()
            .compute(true)
            .no_near_inside(true) // Q8: no interior points to handle
            .no_narrow(true)      // Q10: not a narrow distribution
            .build_from_iter(pts)
            .expect("Failed to compute convex hull");

        let mut facets = Vec::new();
        for simplex in qh.simplices() {
            let vertices: Vec<usize> = simplex
                .vertices()
                .expect("Failed to get vertices")
                .iter()
                .map(|v| v.index(&qh).expect("Failed to get vertex index"))
                .collect();

            if vertices.len() == 3 {
                facets.push(HullFacet {
                    indices: [vertices[0], vertices[1], vertices[2]],
                });
            }
        }

        ConvexHull { facets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hull_tetrahedron() {
        // A tetrahedron has 4 facets
        let points = vec![
            Vec3::new(1.0, 0.0, -1.0 / 2.0_f32.sqrt()),
            Vec3::new(-1.0, 0.0, -1.0 / 2.0_f32.sqrt()),
            Vec3::new(0.0, 1.0, 1.0 / 2.0_f32.sqrt()),
            Vec3::new(0.0, -1.0, 1.0 / 2.0_f32.sqrt()),
        ];
        let hull = ConvexHull::compute(&points);
        assert_eq!(hull.facets.len(), 4);
    }
}
