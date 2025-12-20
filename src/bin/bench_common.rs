use glam::Vec3;
use rand::Rng;

use hex3::geometry::fibonacci_sphere_points_with_rng;

pub fn parse_count(s: &str) -> Result<usize, String> {
    let s = s.to_lowercase();
    let (num_str, multiplier) = if s.ends_with('m') {
        (&s[..s.len() - 1], 1_000_000)
    } else if s.ends_with('k') {
        (&s[..s.len() - 1], 1_000)
    } else {
        (s.as_str(), 1)
    };

    num_str
        .parse::<f64>()
        .map(|n| (n * multiplier as f64) as usize)
        .map_err(|e| format!("Invalid number '{}': {}", s, e))
}

pub fn mean_spacing(num_points: usize) -> f32 {
    if num_points == 0 {
        return 0.0;
    }
    (4.0 * std::f32::consts::PI / num_points as f32).sqrt()
}

pub fn fibonacci_points_with_jitter<R: Rng>(n: usize, jitter_scale: f32, rng: &mut R) -> Vec<Vec3> {
    let jitter = mean_spacing(n) * jitter_scale;
    fibonacci_sphere_points_with_rng(n, jitter, rng)
}

pub fn inject_bad_points<R: Rng>(
    points: &mut Vec<Vec3>,
    count: usize,
    spacing: f32,
    rng: &mut R,
) -> usize {
    if count == 0 || points.is_empty() {
        return 0;
    }

    let min_angle = spacing * 0.4;
    let max_angle = spacing * 0.9;
    let spacing_chord_sq = 2.0 - 2.0 * spacing.cos();
    let max_attempts = count.saturating_mul(200).max(100);
    let mut added = 0usize;
    let mut attempts = 0usize;

    while added < count && attempts < max_attempts {
        attempts += 1;
        let base_idx = rng.gen_range(0..points.len());
        let base = points[base_idx];

        let tangent = random_tangent_vector(base, rng);
        if tangent.length_squared() < 1e-10 {
            continue;
        }
        let angle = rng.gen_range(min_angle..max_angle);
        let candidate = (base * angle.cos() + tangent * angle.sin()).normalize();

        let mut ok = true;
        for (i, p) in points.iter().enumerate() {
            let dist_sq = (*p - candidate).length_squared();
            if dist_sq < spacing_chord_sq && i != base_idx {
                ok = false;
                break;
            }
        }
        if ok {
            points.push(candidate);
            added += 1;
        }
    }

    added
}

fn random_tangent_vector<R: Rng>(p: Vec3, rng: &mut R) -> Vec3 {
    let arbitrary = if p.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = p.cross(arbitrary).normalize();
    let v = p.cross(u);
    let angle: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    u * angle.cos() + v * angle.sin()
}
