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
#[derive(Clone, Debug, PartialEq)]
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

    /// Physical Euler angular velocity in radians per million years.
    pub fn angular_velocity_rad_per_myr(&self) -> f32 {
        self.angular_velocity * MAX_PLATE_ANGULAR_SPEED_RAD_PER_MYR
    }

    /// Physical surface velocity in kilometres per million years.
    pub fn velocity_km_per_myr_at(&self, point: Vec3) -> Vec3 {
        self.velocity_at(point) * MAX_PLATE_SPEED_KM_PER_MYR
    }
}

/// Geological window used by tectonic-history consumers. Keeping it in Dynamics makes
/// exported provenance explicit and permits future per-world clock distributions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TectonicClock {
    pub lookback_myr: f32,
    pub step_myr: f32,
}

impl Default for TectonicClock {
    fn default() -> Self {
        Self {
            lookback_myr: TECTONIC_HISTORY_LOOKBACK_MYR,
            step_myr: TECTONIC_HISTORY_STEP_MYR,
        }
    }
}

/// Plate dynamics: motion for each plate.
pub struct Dynamics {
    /// Euler pole for each plate (rotation axis + angular velocity).
    pub euler_poles: Vec<EulerPole>,
    pub clock: TectonicClock,
}

impl Dynamics {
    /// Generate plate dynamics from plate assignments.
    pub fn generate<R: Rng>(plates: &Plates, rng: &mut R) -> Self {
        let euler_poles = generate_euler_poles(plates.num_plates, rng);
        Self {
            euler_poles,
            clock: TectonicClock::default(),
        }
    }

    /// Get the Euler pole for a plate.
    pub fn euler_pole(&self, plate_id: usize) -> &EulerPole {
        &self.euler_poles[plate_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_plate_speed_has_explicit_physical_conversion() {
        let pole = EulerPole {
            axis: Vec3::Z,
            angular_velocity: 1.0,
        };
        let point = Vec3::X;
        let speed = pole.velocity_km_per_myr_at(point).length();
        assert!((speed - MAX_PLATE_SPEED_KM_PER_MYR).abs() < 1e-5);
        assert!((pole.angular_velocity_rad_per_myr() * PLANET_RADIUS_KM - speed).abs() < 1e-4);
    }

    #[test]
    fn tectonic_clock_is_positive_and_evenly_subdivides() {
        let clock = TectonicClock::default();
        assert!(clock.lookback_myr > 0.0);
        assert!(clock.step_myr > 0.0);
        let steps = clock.lookback_myr / clock.step_myr;
        assert!((steps - steps.round()).abs() < 1e-6);
    }
}

/// Generate random Euler poles for each plate.
fn generate_euler_poles<R: Rng>(num_plates: usize, rng: &mut R) -> Vec<EulerPole> {
    (0..num_plates).map(|_| sample_euler_pole(rng)).collect()
}

/// Draw one Euler vector from the same isotropic prior used at the present.
/// Historical-motion experiments call this at explicit reorganization events
/// so their marginal speed/axis distribution does not introduce a second law.
pub(crate) fn sample_euler_pole<R: Rng>(rng: &mut R) -> EulerPole {
    let theta = rng.gen::<f32>() * std::f32::consts::TAU;
    let phi = (1.0 - 2.0 * rng.gen::<f32>()).acos();
    let axis = Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos());
    let angular_velocity = (rng.gen::<f32>() - 0.5) * 2.0 * MAX_ANGULAR_VELOCITY;
    EulerPole {
        axis,
        angular_velocity,
    }
}
