//! Plate dynamics - Euler poles and plate types.

use glam::Vec3;
use rand::seq::SliceRandom;
use rand::Rng;

use super::constants::*;
use super::Plates;

/// Type of tectonic plate - affects base elevation and collision behavior.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
        self.axis.cross(point) * self.angular_velocity
    }
}

/// Plate dynamics: types and motion for each plate.
pub struct Dynamics {
    /// Type of each plate (continental or oceanic).
    pub plate_types: Vec<PlateType>,

    /// Euler pole for each plate (rotation axis + angular velocity).
    pub euler_poles: Vec<EulerPole>,
}

impl Dynamics {
    /// Generate plate dynamics from plate assignments.
    pub fn generate<R: Rng>(plates: &Plates, rng: &mut R) -> Self {
        let plate_types = assign_plate_types(plates, rng);
        let euler_poles = generate_euler_poles(plates.num_plates, rng);

        Self {
            plate_types,
            euler_poles,
        }
    }

    /// Get the type of a plate.
    pub fn plate_type(&self, plate_id: usize) -> PlateType {
        self.plate_types[plate_id]
    }

    /// Get the Euler pole for a plate.
    pub fn euler_pole(&self, plate_id: usize) -> &EulerPole {
        &self.euler_poles[plate_id]
    }
}

/// Assign plate types to achieve target continental coverage.
fn assign_plate_types<R: Rng>(plates: &Plates, rng: &mut R) -> Vec<PlateType> {
    // Count cells per plate
    let mut plate_sizes: Vec<usize> = vec![0; plates.num_plates];
    for &plate in &plates.cell_plate {
        plate_sizes[plate as usize] += 1;
    }

    // Create list of (plate_id, size, target_size) sorted by target_size descending
    let mut plate_info: Vec<(usize, usize, f32)> = plate_sizes
        .into_iter()
        .enumerate()
        .map(|(id, size)| (id, size, plates.target_sizes[id]))
        .collect();

    plate_info.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Target continental cell count
    let total_cells = plates.cell_plate.len();
    let target = (total_cells as f32 * CONTINENTAL_FRACTION) as usize;

    let mut types = vec![PlateType::Oceanic; plates.num_plates];
    let mut continental_cells = 0;

    // Add randomness: shuffle plates with similar target_size
    let mut i = 0;
    while i < plate_info.len() {
        let base_size = plate_info[i].2;
        let mut j = i + 1;
        while j < plate_info.len() && plate_info[j].2 >= base_size * 0.8 {
            j += 1;
        }
        plate_info[i..j].shuffle(rng);
        i = j;
    }

    // Assign continental until we reach target
    for (plate_id, size, _) in plate_info {
        if continental_cells + size <= target {
            types[plate_id] = PlateType::Continental;
            continental_cells += size;
        }
    }

    types
}

/// Generate random Euler poles for each plate.
fn generate_euler_poles<R: Rng>(num_plates: usize, rng: &mut R) -> Vec<EulerPole> {
    (0..num_plates)
        .map(|_| {
            // Random point on unit sphere for axis
            let theta = rng.gen::<f32>() * std::f32::consts::TAU;
            let phi = (1.0 - 2.0 * rng.gen::<f32>()).acos();
            let axis = Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());

            // Random angular velocity
            let angular_velocity = (rng.gen::<f32>() - 0.5) * 2.0 * MAX_ANGULAR_VELOCITY;

            EulerPole {
                axis,
                angular_velocity,
            }
        })
        .collect()
}
