//! Plate dynamics - Euler poles describing plate motion.
//!
//! Crust type (continental vs oceanic) is a per-cell property of `Crust`,
//! not a plate property: plates are motion units and can carry both crust
//! types (mixed plates are what make passive margins possible).

use glam::Vec3;
use rand::Rng;

use super::constants::*;
use super::Plates;

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

/// Plate dynamics: motion for each plate.
pub struct Dynamics {
    /// Euler pole for each plate (rotation axis + angular velocity).
    pub euler_poles: Vec<EulerPole>,
}

impl Dynamics {
    /// Generate plate dynamics from plate assignments.
    pub fn generate<R: Rng>(plates: &Plates, rng: &mut R) -> Self {
        let euler_poles = generate_euler_poles(plates.num_plates, rng);
        Self { euler_poles }
    }

    /// Get the Euler pole for a plate.
    pub fn euler_pole(&self, plate_id: usize) -> &EulerPole {
        &self.euler_poles[plate_id]
    }
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
