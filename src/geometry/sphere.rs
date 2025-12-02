use glam::Vec3;
use rand::Rng;

/// Generate `n` uniformly distributed random points on a unit sphere.
pub fn random_sphere_points(n: usize) -> Vec<Vec3> {
    let mut rng = rand::thread_rng();
    random_sphere_points_with_rng(n, &mut rng)
}

/// Generate `n` uniformly distributed random points on a unit sphere using a provided RNG.
pub fn random_sphere_points_with_rng<R: Rng>(n: usize, rng: &mut R) -> Vec<Vec3> {
    (0..n)
        .map(|_| {
            // Use spherical coordinates with uniform distribution
            // theta: azimuthal angle [0, 2*PI)
            // phi: polar angle, derived from uniform z in [-1, 1]
            let z: f32 = rng.gen_range(-1.0..1.0);
            let theta: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_points_on_unit_sphere() {
        let points = random_sphere_points(100);
        for p in &points {
            let len = p.length();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "Point not on unit sphere: length = {}",
                len
            );
        }
    }
}
