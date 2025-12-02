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
    /// Reference cell count for stress scaling (constants tuned for this density).
    pub const REFERENCE_NUM_CELLS: f32 = 5000.0;
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
    pub const CONT_MAX_RIFT: f32 = 0.3;

    // Sqrt-based elevation response parameters (oceanic crust)
    // Oceanic crust is thinner and denser - limited uplift, forms islands not mountains
    /// Scale factor for compression → volcanic island height (oceanic).
    pub const OCEAN_COMPRESSION_SENS: f32 = 0.1;
    /// Scale factor for tension → mid-ocean ridge height (oceanic).
    pub const OCEAN_TENSION_SENS: f32 = 0.15;
    /// Maximum uplift from compression (oceanic) - volcanic islands, not mountains.
    pub const OCEAN_MAX_MOUNTAIN: f32 = 0.30;
    /// Maximum mid-ocean ridge height from tension (oceanic) - UPLIFT, not rift.
    pub const OCEAN_MAX_RIDGE: f32 = 0.15;

    // 6-way plate interaction multipliers
    // CONVERGENT - plates pushing together
    /// Cont+Cont: Himalayas-scale, highest mountains (neither subducts)
    pub const CONV_CONT_CONT: f32 = 1.5;
    /// Ocean+Ocean: Volcanic island arc (Japan, Philippines) - modest uplift
    pub const CONV_OCEAN_OCEAN: f32 = 0.4;
    /// Cont side of Cont+Ocean: Andes-style coastal mountains
    pub const CONV_CONT_OCEAN: f32 = 1.2;
    /// Ocean side of Cont+Ocean: Trench (Mariana, Peru-Chile) - depression
    pub const CONV_OCEAN_CONT: f32 = -0.5;

    // DIVERGENT - plates pulling apart (all negative → routes to tension field)
    /// Cont+Cont: East African Rift, Red Sea - significant depression
    pub const DIV_CONT_CONT: f32 = -0.6;
    /// Ocean+Ocean: Mid-Atlantic Ridge - tension that causes uplift (handled in stress_to_elevation)
    pub const DIV_OCEAN_OCEAN: f32 = -0.5;
    /// Cont side of Cont+Ocean divergence: mild stretching, continental crust stays buoyant
    pub const DIV_CONT_OCEAN: f32 = -0.15;
    /// Ocean side of Cont+Ocean divergence: ridge formation, similar to ocean-ocean
    pub const DIV_OCEAN_CONT: f32 = -0.5;
}

/// Assign plate types to achieve target continental coverage by cell count.
///
/// Shuffles plates randomly, then greedily assigns as continental until
/// the target fraction is reached. This ensures predictable land/ocean ratio
/// while maintaining randomness in which plates become continental.
pub fn assign_plate_types_by_coverage_with_rng<R: Rng>(
    cell_plate: &[u32],
    num_plates: usize,
    rng: &mut R,
) -> Vec<PlateType> {
    use rand::seq::SliceRandom;

    // Count cells per plate
    let mut plate_sizes: Vec<usize> = vec![0; num_plates];
    for &plate in cell_plate {
        plate_sizes[plate as usize] += 1;
    }

    // Create shuffled list of (plate_id, size)
    let mut plates: Vec<(usize, usize)> = plate_sizes.into_iter().enumerate().collect();
    plates.shuffle(rng);

    // Greedily assign plates as continental until we reach target
    let total_cells = cell_plate.len();
    let target = (total_cells as f32 * constants::CONTINENTAL_FRACTION) as usize;

    let mut types = vec![PlateType::Oceanic; num_plates];
    let mut continental_cells = 0;

    for (plate_id, size) in plates {
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
pub fn calculate_boundary_stress(
    adjacency: &[Vec<usize>],
    cell_plate: &[u32],
    euler_poles: &[EulerPole],
    plate_types: &[PlateType],
    voronoi: &SphericalVoronoi,
) -> Vec<f32> {
    use PlateType::*;

    let num_cells = cell_plate.len();
    let mut stress = vec![0.0f32; num_cells];

    for cell_idx in 0..num_cells {
        let plate_a = cell_plate[cell_idx] as usize;
        let cell_pos = voronoi.generators[cell_idx];
        let type_a = plate_types[plate_a];

        for &neighbor_idx in &adjacency[cell_idx] {
            let plate_b = cell_plate[neighbor_idx] as usize;

            // Only calculate stress at plate boundaries
            if plate_a != plate_b {
                let neighbor_pos = voronoi.generators[neighbor_idx];
                let type_b = plate_types[plate_b];

                // Calculate velocities at the midpoint of the boundary
                let boundary_point = (cell_pos + neighbor_pos).normalize();
                let vel_a = euler_poles[plate_a].velocity_at(boundary_point);
                let vel_b = euler_poles[plate_b].velocity_at(boundary_point);

                // Relative velocity
                let relative_vel = vel_a - vel_b;

                // Boundary normal points from cell A toward cell B
                let boundary_normal = (neighbor_pos - cell_pos).normalize();

                // If A moves toward B (relative_vel aligns with normal) → positive (convergent)
                // If A moves away from B → negative (divergent)
                let convergence = relative_vel.dot(boundary_normal);

                // Apply 6-way plate interaction logic
                let stress_contribution = if convergence > 0.0 {
                    // CONVERGENT - plates pushing together
                    match (type_a, type_b) {
                        (Continental, Continental) => convergence * constants::CONV_CONT_CONT,
                        (Oceanic, Oceanic) => convergence * constants::CONV_OCEAN_OCEAN,
                        (Continental, Oceanic) => convergence * constants::CONV_CONT_OCEAN,
                        (Oceanic, Continental) => convergence * constants::CONV_OCEAN_CONT,
                    }
                } else {
                    // DIVERGENT - plates pulling apart
                    let divergence = -convergence;
                    match (type_a, type_b) {
                        (Continental, Continental) => divergence * constants::DIV_CONT_CONT,
                        (Oceanic, Oceanic) => divergence * constants::DIV_OCEAN_OCEAN,
                        (Continental, Oceanic) => divergence * constants::DIV_CONT_OCEAN,
                        (Oceanic, Continental) => divergence * constants::DIV_OCEAN_CONT,
                    }
                };

                stress[cell_idx] += stress_contribution;
            }
        }
    }

    // Scale stress to be independent of cell density.
    // More cells = smaller cells = each contributes less stress.
    // sqrt because stress scales with edge length (linear), not area.
    let density_scale = (num_cells as f32 / constants::REFERENCE_NUM_CELLS).sqrt();
    for s in &mut stress {
        *s /= density_scale;
    }

    stress
}

/// Propagate stress from boundary cells using plate-constrained sum-decay model.
///
/// Returns (compression, tension) where:
/// - compression = sum of (convergent_stress × decay) from same-plate boundaries
/// - tension = sum of (divergent_stress × decay) from same-plate boundaries
///
/// Key properties:
/// - Stress only propagates within the same plate (no cross-plate bleeding)
/// - Multiple boundary cells contribute additively (physically correct superposition)
/// - Already density-independent: boundary stress is pre-normalized in calculate_boundary_stress
pub fn propagate_stress(
    boundary_stress: &[f32],
    cell_plate: &[u32],
    voronoi: &SphericalVoronoi,
) -> (Vec<f32>, Vec<f32>) {
    let num_cells = boundary_stress.len();
    let mut compression = vec![0.0f32; num_cells];
    let mut tension = vec![0.0f32; num_cells];

    // Determine number of plates
    let num_plates = cell_plate.iter().map(|&p| p as usize).max().unwrap_or(0) + 1;

    // Build map: plate_id -> Vec of boundary cell indices on that plate
    let mut plate_boundaries: Vec<Vec<usize>> = vec![Vec::new(); num_plates];
    for (cell_idx, &stress) in boundary_stress.iter().enumerate() {
        if stress.abs() > 0.001 {
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

            let stress = boundary_stress[boundary_idx];
            if stress > 0.0 {
                // Convergent boundary → compression
                compression[cell_idx] += stress * decay;
            } else {
                // Divergent boundary → tension (store as positive value)
                tension[cell_idx] += -stress * decay;
            }
        }
    }

    (compression, tension)
}

/// Convert compression and tension to elevation using sqrt response.
///
/// Uses sqrt for diminishing returns that never truly plateaus:
/// - Rises quickly at low stress (responsive to small forces)
/// - Gradually slows at high stress (diminishing returns)
/// - No hard saturation - highest stress = highest peaks
///
/// Continental and oceanic crust respond differently:
/// - Continental: thick, buoyant → can form high mountains, tension causes rifts
/// - Oceanic: thin, dense → limited uplift, but tension causes mid-ocean ridges (UPLIFT)
pub fn stress_to_elevation(compression: f32, tension: f32, plate_type: PlateType) -> f32 {
    match plate_type {
        PlateType::Continental => {
            // Continental: compression → mountains, tension → rifts
            let mountain = (compression * constants::CONT_COMPRESSION_SENS)
                .sqrt()
                .min(constants::CONT_MAX_MOUNTAIN);
            let rift = (tension * constants::CONT_TENSION_SENS)
                .sqrt()
                .min(constants::CONT_MAX_RIFT);
            constants::CONTINENTAL_BASE + mountain - rift
        }
        PlateType::Oceanic => {
            // Oceanic: compression → volcanic islands, tension → mid-ocean ridges
            // Use max to prevent double-dipping at triple junctions
            let island = (compression * constants::OCEAN_COMPRESSION_SENS)
                .sqrt()
                .min(constants::OCEAN_MAX_MOUNTAIN);
            let ridge = (tension * constants::OCEAN_TENSION_SENS)
                .sqrt()
                .min(constants::OCEAN_MAX_RIDGE);
            constants::OCEANIC_BASE + island.max(ridge)
        }
    }
}

/// Generate heightmap from compression, tension, and plate types.
pub fn generate_heightmap(
    compression: &[f32],
    tension: &[f32],
    cell_plate: &[u32],
    plate_types: &[PlateType],
) -> Vec<f32> {
    compression
        .iter()
        .zip(tension.iter())
        .enumerate()
        .map(|(i, (&c, &t))| {
            let plate = cell_plate[i] as usize;
            stress_to_elevation(c, t, plate_types[plate])
        })
        .collect()
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
