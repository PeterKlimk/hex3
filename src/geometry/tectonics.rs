use glam::Vec3;
use rand::Rng;

use super::SphericalVoronoi;

/// Type of tectonic plate - affects base elevation and collision behavior.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PlateType {
    Continental,
    Oceanic,
}

/// Euler pole describing plate rotation on a sphere.
#[derive(Clone, Debug)]
pub struct EulerPole {
    /// Normalized axis of rotation (point on unit sphere).
    pub axis: Vec3,
    /// Angular velocity (radians per unit time, sign indicates direction).
    pub angular_velocity: f32,
}

impl EulerPole {
    /// Calculate the velocity vector at a given point on the sphere.
    ///
    /// Velocity is tangent to the sphere, perpendicular to the great circle
    /// through the Euler pole.
    pub fn velocity_at(&self, point: Vec3) -> Vec3 {
        // v = ω × r, where ω is angular velocity vector and r is position
        // The magnitude scales with sin(angle from pole)
        self.axis.cross(point) * self.angular_velocity
    }
}

/// Constants for tectonic simulation.
pub mod constants {
    /// Target fraction of surface area that should be continental.
    pub const CONTINENTAL_FRACTION: f32 = 0.30;
    /// Global stress scale factor (compensates for edge-length weighting).
    /// Multipliers below are relative ratios; this controls overall magnitude.
    pub const STRESS_SCALE: f32 = 25.0;
    /// How far stress propagates from boundaries (in radians).
    /// 0.07 radians ≈ 4° ≈ 450km on Earth scale.
    pub const STRESS_DECAY_LENGTH: f32 = 0.04;
    /// Base elevation for continental plates.
    pub const CONTINENTAL_BASE: f32 = 0.05;
    /// Base elevation for oceanic plates.
    pub const OCEANIC_BASE: f32 = -0.2;
    /// Angular velocity range for random Euler poles.
    pub const MAX_ANGULAR_VELOCITY: f32 = 1.0;

    // Sqrt-based elevation response parameters (continental crust)
    /// Scale factor for compression → mountain height (continental).
    pub const CONT_COMPRESSION_SENS: f32 = 0.4;
    /// Scale factor for tension → rift depth (continental).
    pub const CONT_TENSION_SENS: f32 = 0.3;
    /// Maximum mountain height from compression (continental).
    pub const CONT_MAX_MOUNTAIN: f32 = 0.8;
    /// Maximum rift depth from tension (continental).
    /// Capped so continental crust stays above oceanic baseline (-0.2).
    pub const CONT_MAX_RIFT: f32 = 0.2;

    // Oceanic crust - shared sensitivity, asymmetric caps
    // Both compression and tension cause uplift, but with different ceilings:
    // - Compression (volcanic arcs): can build islands
    // - Tension (ridges): limited by isostatic equilibrium, stays underwater
    /// Scale factor for stress → oceanic uplift.
    pub const OCEAN_SENSITIVITY: f32 = 0.12;
    /// Maximum oceanic uplift from compression (volcanic edifice - can create islands).
    pub const OCEAN_COMPRESSION_MAX: f32 = 0.25;
    /// Maximum oceanic uplift from tension (isostatic limit - stays underwater).
    pub const OCEAN_TENSION_MAX: f32 = 0.12;

    // 6-way plate interaction multipliers (all positive - sign comes from convergence)
    // CONVERGENT - plates pushing together
    /// Cont+Cont: Himalayas-scale, highest mountains (neither subducts)
    pub const CONV_CONT_CONT: f32 = 1.5;
    /// Ocean+Ocean: Volcanic island arc (Japan, Philippines) - modest uplift
    pub const CONV_OCEAN_OCEAN: f32 = 0.4;
    /// Cont side of Cont+Ocean: Andes-style coastal mountains
    pub const CONV_CONT_OCEAN: f32 = 1.2;
    /// Ocean side of Cont+Ocean: minimal - subducting plate goes down, not up
    pub const CONV_OCEAN_CONT: f32 = 0.1;

    // DIVERGENT - plates pulling apart (positive magnitudes - sign from convergence)
    /// Cont+Cont: East African Rift, Red Sea
    pub const DIV_CONT_CONT: f32 = 0.6;
    /// Ocean+Ocean: Mid-Atlantic Ridge
    pub const DIV_OCEAN_OCEAN: f32 = 0.5;
    /// Cont side of Cont+Ocean: modest rifting at passive margin
    pub const DIV_CONT_OCEAN: f32 = 0.1;
    /// Ocean side of Cont+Ocean: thermal uplift near margin
    pub const DIV_OCEAN_CONT: f32 = 0.3;

    // Plate generation tuning
    /// Fraction of ideal seed spacing to use as minimum distance (0.5 = half ideal).
    pub const SEED_SPACING_FRACTION: f32 = 0.5;
    /// Log-normal spread for plate target sizes. Higher = more size variance.
    pub const TARGET_SIZE_SIGMA: f32 = 0.4;
    /// Maximum ratio between largest and smallest plate target size.
    pub const TARGET_SIZE_MAX_RATIO: f32 = 4.0;

    // Relief rendering
    /// Scale factor for elevation displacement in relief view.
    /// 0.1 means max elevation (1.0) displaces vertex by 10% of sphere radius.
    pub const RELIEF_SCALE: f32 = 0.1;
    /// Weight of noise vs distance in cell priority (0.0 = pure distance, 1.0 = pure noise).
    pub const NOISE_WEIGHT: f32 = 1.0;
    /// Bonus per same-plate neighbor when claiming a cell (encourages compact shapes).
    pub const NEIGHBOR_BONUS: f32 = 0.1;
    /// Base frequency for fBm noise on sphere.
    pub const NOISE_FREQUENCY: f64 = 2.0;
    /// Number of octaves for fBm noise.
    pub const NOISE_OCTAVES: usize = 4;

    // Elevation noise
    /// Additional noise amplitude per unit of |stress| (active areas get more noise)
    pub const ELEVATION_NOISE_STRESS: f32 = 0.2;
    /// Base noise amplitude for continental plates
    pub const ELEVATION_NOISE_CONTINENTAL: f32 = 0.1;
    /// Base noise amplitude for oceanic plates
    pub const ELEVATION_NOISE_OCEANIC: f32 = 0.05;
    /// Base frequency for elevation noise
    pub const ELEVATION_NOISE_FREQUENCY: f64 = 16.0;
    /// Number of octaves for elevation fBm
    pub const ELEVATION_NOISE_OCTAVES: usize = 4;
}

/// Assign plate types to achieve target continental coverage by cell count.
///
/// Larger plates (by target_size) are more likely to become continental.
/// Uses weighted random selection to introduce variety while favoring big plates.
pub fn assign_plate_types_by_coverage_with_rng<R: Rng>(
    cell_plate: &[u32],
    num_plates: usize,
    target_sizes: &[f32],
    rng: &mut R,
) -> Vec<PlateType> {
    use rand::seq::SliceRandom;

    // Count cells per plate
    let mut plate_sizes: Vec<usize> = vec![0; num_plates];
    for &plate in cell_plate {
        plate_sizes[plate as usize] += 1;
    }

    // Create list of (plate_id, size, target_size) sorted by target_size descending
    let mut plates: Vec<(usize, usize, f32)> = plate_sizes
        .into_iter()
        .enumerate()
        .map(|(id, size)| (id, size, target_sizes[id]))
        .collect();

    // Sort by target_size descending (larger plates first)
    plates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Greedily assign plates as continental until we reach target
    // Larger plates get priority, with some randomness via shuffle within similar sizes
    let total_cells = cell_plate.len();
    let target = (total_cells as f32 * constants::CONTINENTAL_FRACTION) as usize;

    let mut types = vec![PlateType::Oceanic; num_plates];
    let mut continental_cells = 0;

    // Add some randomness: shuffle plates with similar target_size (within 20%)
    let mut i = 0;
    while i < plates.len() {
        let base_size = plates[i].2;
        let mut j = i + 1;
        while j < plates.len() && plates[j].2 >= base_size * 0.8 {
            j += 1;
        }
        // Shuffle plates[i..j] which have similar target_size
        plates[i..j].shuffle(rng);
        i = j;
    }

    for (plate_id, size, _) in plates {
        if continental_cells + size <= target {
            types[plate_id] = PlateType::Continental;
            continental_cells += size;
        }
    }

    types
}

/// Generate random Euler poles for each plate.
pub fn generate_euler_poles(num_plates: usize) -> Vec<EulerPole> {
    let mut rng = rand::thread_rng();
    generate_euler_poles_with_rng(num_plates, &mut rng)
}

/// Generate random Euler poles using a provided RNG.
pub fn generate_euler_poles_with_rng<R: Rng>(num_plates: usize, rng: &mut R) -> Vec<EulerPole> {
    (0..num_plates)
        .map(|_| {
            // Random point on unit sphere for axis
            let theta = rng.gen::<f32>() * std::f32::consts::TAU;
            let phi = (1.0 - 2.0 * rng.gen::<f32>()).acos();
            let axis = Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());
            // Random angular velocity
            let angular_velocity = (rng.gen::<f32>() - 0.5) * 2.0 * constants::MAX_ANGULAR_VELOCITY;
            EulerPole {
                axis,
                angular_velocity,
            }
        })
        .collect()
}

/// Calculate boundary stress for each cell.
///
/// Stress is calculated only at plate boundaries based on relative velocity
/// between adjacent plates. Different plate type combinations produce different
/// stress responses (6 distinct interactions: 3 plate pairings × 2 directions).
///
/// Each edge contribution is weighted by its arc length, making the result
/// naturally independent of cell density (no additional scaling needed).
pub fn calculate_boundary_stress(
    adjacency: &[Vec<usize>],
    cell_plate: &[u32],
    euler_poles: &[EulerPole],
    plate_types: &[PlateType],
    voronoi: &SphericalVoronoi,
) -> Vec<f32> {
    use std::collections::HashSet;
    use PlateType::*;

    let num_cells = cell_plate.len();
    let mut stress = vec![0.0f32; num_cells];

    for cell_idx in 0..num_cells {
        let plate_a = cell_plate[cell_idx] as usize;
        let cell_pos = voronoi.generators[cell_idx];
        let type_a = plate_types[plate_a];
        let cell_verts: HashSet<usize> = voronoi.cells[cell_idx]
            .vertex_indices
            .iter()
            .copied()
            .collect();

        for &neighbor_idx in &adjacency[cell_idx] {
            let plate_b = cell_plate[neighbor_idx] as usize;

            // Only calculate stress at plate boundaries
            if plate_a != plate_b {
                let neighbor_pos = voronoi.generators[neighbor_idx];
                let type_b = plate_types[plate_b];

                // Find shared edge vertices and compute arc length
                let neighbor_verts: HashSet<usize> = voronoi.cells[neighbor_idx]
                    .vertex_indices
                    .iter()
                    .copied()
                    .collect();
                let shared: Vec<usize> =
                    cell_verts.intersection(&neighbor_verts).copied().collect();

                let edge_length = if shared.len() == 2 {
                    let v0 = voronoi.vertices[shared[0]];
                    let v1 = voronoi.vertices[shared[1]];
                    // Arc length on unit sphere = angle between points
                    v0.dot(v1).clamp(-1.0, 1.0).acos()
                } else {
                    // Fallback: shouldn't happen for valid Voronoi adjacency
                    0.1
                };

                // Calculate velocities at the midpoint of the boundary
                let boundary_point = (cell_pos + neighbor_pos).normalize();
                let vel_a = euler_poles[plate_a].velocity_at(boundary_point);
                let vel_b = euler_poles[plate_b].velocity_at(boundary_point);

                // Relative velocity (already tangent to sphere at boundary_point)
                let relative_vel = vel_a - vel_b;

                // Boundary direction: project chord into tangent plane at boundary_point
                let chord = neighbor_pos - cell_pos;
                let tangent_normal = chord - boundary_point * chord.dot(boundary_point);
                let tangent_normal = if tangent_normal.length_squared() > 1e-10 {
                    tangent_normal.normalize()
                } else {
                    // Degenerate case: cells are antipodal, use arbitrary tangent
                    let up = if boundary_point.y.abs() < 0.9 {
                        Vec3::Y
                    } else {
                        Vec3::X
                    };
                    boundary_point.cross(up).normalize()
                };

                // If A moves toward B (relative_vel aligns with tangent_normal) → positive (convergent)
                // If A moves away from B → negative (divergent)
                let convergence = relative_vel.dot(tangent_normal);

                // Apply 6-way plate interaction logic
                // All multipliers are positive magnitudes; sign comes from convergence
                let multiplier = if convergence > 0.0 {
                    // CONVERGENT - plates pushing together
                    match (type_a, type_b) {
                        (Continental, Continental) => constants::CONV_CONT_CONT,
                        (Oceanic, Oceanic) => constants::CONV_OCEAN_OCEAN,
                        (Continental, Oceanic) => constants::CONV_CONT_OCEAN,
                        (Oceanic, Continental) => constants::CONV_OCEAN_CONT,
                    }
                } else {
                    // DIVERGENT - plates pulling apart
                    match (type_a, type_b) {
                        (Continental, Continental) => constants::DIV_CONT_CONT,
                        (Oceanic, Oceanic) => constants::DIV_OCEAN_OCEAN,
                        (Continental, Oceanic) => constants::DIV_CONT_OCEAN,
                        (Oceanic, Continental) => constants::DIV_OCEAN_CONT,
                    }
                };

                // Weight by edge length for density-independent stress
                stress[cell_idx] +=
                    convergence * multiplier * edge_length * constants::STRESS_SCALE;
            }
        }
    }

    stress
}

/// Propagate stress from boundary cells using plate-constrained sum-decay model.
///
/// Returns a single stress field where:
/// - Positive values = compression (from convergent boundaries)
/// - Negative values = tension (from divergent boundaries)
///
/// Key properties:
/// - Stress only propagates within the same plate (no cross-plate bleeding)
/// - Multiple boundary cells contribute additively (physically correct superposition)
/// - Already density-independent: boundary stress is pre-normalized in calculate_boundary_stress
pub fn propagate_stress(
    boundary_stress: &[f32],
    cell_plate: &[u32],
    voronoi: &SphericalVoronoi,
) -> Vec<f32> {
    let num_cells = boundary_stress.len();
    let mut stress = vec![0.0f32; num_cells];

    // Determine number of plates
    let num_plates = cell_plate.iter().map(|&p| p as usize).max().unwrap_or(0) + 1;

    // Build map: plate_id -> Vec of boundary cell indices on that plate
    let mut plate_boundaries: Vec<Vec<usize>> = vec![Vec::new(); num_plates];
    for (cell_idx, &s) in boundary_stress.iter().enumerate() {
        if s.abs() > 0.001 {
            let plate_id = cell_plate[cell_idx] as usize;
            plate_boundaries[plate_id].push(cell_idx);
        }
    }

    // For each cell, sum decayed stress from same-plate boundary cells
    for cell_idx in 0..num_cells {
        let plate_id = cell_plate[cell_idx] as usize;
        let cell_pos = voronoi.generators[cell_idx];

        // Only iterate over boundary cells on the SAME plate
        for &boundary_idx in &plate_boundaries[plate_id] {
            let boundary_pos = voronoi.generators[boundary_idx];
            // Great circle distance
            let distance = cell_pos.dot(boundary_pos).clamp(-1.0, 1.0).acos();
            let decay = (-distance / constants::STRESS_DECAY_LENGTH).exp();

            // Accumulate with sign preserved
            stress[cell_idx] += boundary_stress[boundary_idx] * decay;
        }
    }

    stress
}

/// Convert stress to elevation using sqrt response.
///
/// Uses sqrt for diminishing returns that never truly plateaus:
/// - Rises quickly at low stress (responsive to small forces)
/// - Gradually slows at high stress (diminishing returns)
/// - No hard saturation - highest stress = highest peaks
///
/// Continental and oceanic crust respond differently:
/// - Continental: asymmetric (compression → mountains, tension → rifts)
/// - Oceanic: symmetric uplift (both compression and tension → uplift)
pub fn stress_to_elevation(stress: f32, plate_type: PlateType) -> f32 {
    match plate_type {
        PlateType::Continental => {
            // Asymmetric response: compression → mountains, tension → rifts
            let (sens, max) = if stress >= 0.0 {
                (
                    constants::CONT_COMPRESSION_SENS,
                    constants::CONT_MAX_MOUNTAIN,
                )
            } else {
                (constants::CONT_TENSION_SENS, constants::CONT_MAX_RIFT)
            };
            let effect = (stress.abs() * sens).sqrt().min(max);
            constants::CONTINENTAL_BASE + effect * stress.signum()
        }
        PlateType::Oceanic => {
            // Both compression and tension cause uplift, but with different ceilings:
            // - Compression: volcanic edifices can build above sea level
            // - Tension: isostatic equilibrium limits ridges to stay underwater
            let max = if stress >= 0.0 {
                constants::OCEAN_COMPRESSION_MAX
            } else {
                constants::OCEAN_TENSION_MAX
            };
            let effect = (stress.abs() * constants::OCEAN_SENSITIVITY)
                .sqrt()
                .min(max);
            constants::OCEANIC_BASE + effect
        }
    }
}

/// Generate heightmap from stress and plate types.
pub fn generate_heightmap(
    cell_stress: &[f32],
    cell_plate: &[u32],
    plate_types: &[PlateType],
) -> Vec<f32> {
    cell_stress
        .iter()
        .enumerate()
        .map(|(i, &stress)| {
            let plate = cell_plate[i] as usize;
            stress_to_elevation(stress, plate_types[plate])
        })
        .collect()
}

/// Generate heightmap with fBm noise modulated by stress and plate type.
/// Returns (elevations, noise_contributions) for visualization.
pub fn generate_heightmap_with_noise(
    cell_stress: &[f32],
    cell_plate: &[u32],
    plate_types: &[PlateType],
    voronoi: &super::SphericalVoronoi,
    elevation_fbm: &noise::Fbm<noise::Perlin>,
) -> (Vec<f32>, Vec<f32>) {
    use noise::NoiseFn;

    let mut elevations = Vec::with_capacity(cell_stress.len());
    let mut noise_contributions = Vec::with_capacity(cell_stress.len());

    for (i, &stress) in cell_stress.iter().enumerate() {
        let plate_type = plate_types[cell_plate[i] as usize];

        // Base elevation from stress
        let base = stress_to_elevation(stress, plate_type);

        // Noise amplitude: plate type base + stress contribution
        let type_base = match plate_type {
            PlateType::Continental => constants::ELEVATION_NOISE_CONTINENTAL,
            PlateType::Oceanic => constants::ELEVATION_NOISE_OCEANIC,
        };
        let amplitude = type_base + constants::ELEVATION_NOISE_STRESS * stress.abs();

        // Sample fBm at cell position
        let pos = voronoi.generators[i] * constants::ELEVATION_NOISE_FREQUENCY as f32;
        let noise = elevation_fbm.get([pos.x as f64, pos.y as f64, pos.z as f64]) as f32;

        let noise_contribution = noise * amplitude;
        elevations.push(base + noise_contribution);
        noise_contributions.push(noise_contribution);
    }

    (elevations, noise_contributions)
}

/// Convert elevation to color using hypsometric tinting.
pub fn elevation_to_color(elevation: f32) -> Vec3 {
    // Clamp to reasonable range
    let e = elevation.clamp(-0.5, 1.5);

    if e < -0.15 {
        // Deep ocean: dark blue
        let t = (e + 0.5) / 0.35;
        Vec3::new(0.0, 0.1, 0.3).lerp(Vec3::new(0.1, 0.2, 0.5), t)
    } else if e < 0.0 {
        // Shallow ocean/coast: lighter blue
        let t = (e + 0.15) / 0.15;
        Vec3::new(0.1, 0.2, 0.5).lerp(Vec3::new(0.2, 0.4, 0.6), t)
    } else if e < 0.15 {
        // Coastal lowlands: green
        let t = e / 0.15;
        Vec3::new(0.2, 0.5, 0.3).lerp(Vec3::new(0.3, 0.55, 0.3), t)
    } else if e < 0.4 {
        // Midlands: yellow-green to tan
        let t = (e - 0.15) / 0.25;
        Vec3::new(0.3, 0.55, 0.3).lerp(Vec3::new(0.6, 0.5, 0.35), t)
    } else if e < 0.7 {
        // Highlands: brown
        let t = (e - 0.4) / 0.3;
        Vec3::new(0.6, 0.5, 0.35).lerp(Vec3::new(0.5, 0.4, 0.3), t)
    } else {
        // Mountains: gray to white
        let t = ((e - 0.7) / 0.8).clamp(0.0, 1.0);
        Vec3::new(0.5, 0.4, 0.3).lerp(Vec3::new(1.0, 1.0, 1.0), t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_velocity() {
        // Pole at north, point at equator should give eastward velocity
        let pole = EulerPole {
            axis: Vec3::Z,
            angular_velocity: 1.0,
        };
        let point = Vec3::X;
        let vel = pole.velocity_at(point);

        // Velocity should be tangent (perpendicular to point)
        assert!(vel.dot(point).abs() < 0.001);
        // Should be roughly in Y direction for this configuration
        assert!(vel.y.abs() > 0.9);
    }

    #[test]
    fn test_elevation_color_range() {
        // Test that colors are valid RGB
        for e in [-0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.2] {
            let color = elevation_to_color(e);
            assert!(color.x >= 0.0 && color.x <= 1.0);
            assert!(color.y >= 0.0 && color.y <= 1.0);
            assert!(color.z >= 0.0 && color.z <= 1.0);
        }
    }
}
