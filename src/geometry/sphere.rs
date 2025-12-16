use glam::Vec3;
use rand::Rng;

/// Golden ratio for Fibonacci lattice.
const PHI: f32 = 1.618033988749895;

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

/// Generate `n` points on a unit sphere using Fibonacci lattice with jitter.
///
/// Fibonacci lattice produces a near-optimal uniform distribution without iteration.
/// Adding jitter breaks up the spiral pattern for a more organic appearance.
pub fn fibonacci_sphere_points_with_rng<R: Rng>(n: usize, jitter: f32, rng: &mut R) -> Vec<Vec3> {
    use std::f32::consts::TAU;

    (0..n)
        .map(|i| {
            // Fibonacci lattice formula
            let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
            let r = (1.0 - y * y).sqrt();
            let theta = TAU * i as f32 / PHI;

            let mut p = Vec3::new(r * theta.cos(), y, r * theta.sin());

            // Add tangential jitter for organic look
            if jitter > 0.0 {
                let tangent = random_tangent_vector(p, rng);
                p = (p + tangent * jitter).normalize();
            }

            p
        })
        .collect()
}

/// Generate a random unit vector tangent to the sphere at point `p`.
fn random_tangent_vector<R: Rng>(p: Vec3, rng: &mut R) -> Vec3 {
    // Find a vector not parallel to p
    let arbitrary = if p.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };

    // Create orthonormal basis on tangent plane
    let u = p.cross(arbitrary).normalize();
    let v = p.cross(u);

    // Random angle in tangent plane
    let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    u * angle.cos() + v * angle.sin()
}

/// Apply Lloyd relaxation using Voronoi cell vertex centroids.
///
/// This computes the actual Voronoi diagram and moves each point toward
/// the centroid of its Voronoi cell vertices. More expensive than k-NN
/// approximation but produces correct Lloyd behavior.
pub fn lloyd_relax_voronoi(points: &mut [Vec3], iterations: usize) {
    use super::SphericalVoronoi;

    let n = points.len();
    if n < 2 {
        return;
    }

    for _ in 0..iterations {
        let voronoi = SphericalVoronoi::compute(points);

        // Move each point toward its cell's vertex centroid
        for i in 0..voronoi.num_cells() {
            let cell = voronoi.cell(i);
            if cell.is_empty() {
                continue;
            }

            // Compute centroid of cell vertices (spherical mean)
            let mut sum = Vec3::ZERO;
            for &vi in cell.vertex_indices {
                sum += voronoi.vertices[vi];
            }

            // Normalize to project back onto sphere
            if sum.length_squared() > 1e-10 {
                points[i] = sum.normalize();
            }
        }
    }
}

/// Apply Lloyd relaxation using k-means style sampling approximation.
///
/// Instead of computing the full Voronoi diagram, this:
/// 1. Samples random points on the sphere
/// 2. Assigns each sample to its nearest site (using kd-tree)
/// 3. Moves each site to the centroid of its assigned samples
///
/// This approximates Voronoi cell centroids and is faster than full Voronoi
/// for large point sets when samples_per_site is small (e.g., 10-20).
pub fn lloyd_relax_kmeans<R: Rng>(
    points: &mut [Vec3],
    iterations: usize,
    samples_per_site: usize,
    rng: &mut R,
) {
    use kiddo::{ImmutableKdTree, SquaredEuclidean};
    use rayon::prelude::*;

    let n = points.len();
    if n < 2 {
        return;
    }

    let num_samples = n * samples_per_site;

    // Pre-generate all random samples (RNG is sequential)
    let samples: Vec<Vec3> = (0..num_samples)
        .map(|_| {
            let z: f32 = rng.gen_range(-1.0..1.0);
            let theta: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            let r = (1.0 - z * z).sqrt();
            Vec3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect();

    // Pre-allocate entries buffer
    let mut entries: Vec<[f32; 3]> = vec![[0.0; 3]; n];

    // Chunk size: one chunk per thread (not per sample)
    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = num_samples.div_ceil(num_threads);

    for _ in 0..iterations {
        // Build kd-tree from current sites
        for (i, p) in points.iter().enumerate() {
            entries[i] = [p.x, p.y, p.z];
        }
        let tree: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(&entries);

        // Parallel query: each thread processes a large chunk with its own accumulators
        let (sums, counts) = samples
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_sums = vec![Vec3::ZERO; n];
                let mut local_counts = vec![0usize; n];
                for sample in chunk {
                    let query = [sample.x, sample.y, sample.z];
                    let nearest = tree.approx_nearest_one::<SquaredEuclidean>(&query);
                    let site_idx = nearest.item as usize;
                    local_sums[site_idx] += *sample;
                    local_counts[site_idx] += 1;
                }
                (local_sums, local_counts)
            })
            .reduce(
                || (vec![Vec3::ZERO; n], vec![0usize; n]),
                |(mut sums_a, mut counts_a), (sums_b, counts_b)| {
                    for i in 0..n {
                        sums_a[i] += sums_b[i];
                        counts_a[i] += counts_b[i];
                    }
                    (sums_a, counts_a)
                },
            );

        // Move each site to centroid of its samples
        for i in 0..n {
            if counts[i] > 0 {
                let centroid = sums[i] / counts[i] as f32;
                if centroid.length_squared() > 1e-10 {
                    points[i] = centroid.normalize();
                }
            }
        }
    }
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

    #[test]
    fn test_fibonacci_lloyd_normalized() {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let mut points = fibonacci_sphere_points_with_rng(1000, 0.1, &mut rng);
        lloyd_relax_kmeans(&mut points, 2, 20, &mut rng);

        let mut max_deviation = 0.0f32;
        for p in &points {
            let deviation = (p.length() - 1.0).abs();
            max_deviation = max_deviation.max(deviation);
            assert!(
                deviation < 1e-6,
                "Point not normalized after Lloyd: length = {}",
                p.length()
            );
        }
        println!("Max normalization deviation: {:.2e}", max_deviation);
    }
}
