//! Prescribed meridional overturning circulation.
//!
//! A latitude-only streamfunction defines the mean surface meridional flow,
//! its Coriolis-turned zonal component, analytic vertical motion, and a
//! visualization-quality upper return flow.

use glam::Vec3;

use super::constants::*;

#[derive(Debug, Clone, Copy)]
struct CirculationCell {
    phi0: f32,
    phi1: f32,
    amplitude: f32,
    sign: f32,
}

impl CirculationCell {
    fn contains(self, phi: f32) -> bool {
        phi >= self.phi0 - 1e-6 && phi <= self.phi1 + 1e-6
    }

    fn width(self) -> f32 {
        self.phi1 - self.phi0
    }

    fn phase(self, phi: f32) -> f32 {
        std::f32::consts::PI * (phi - self.phi0) / self.width()
    }

    /// Streamfunction: sine arch over the cell, scaled by cos(phi) because
    /// transport across a latitude circle scales with its circumference.
    /// Without the cos factor, v = psi/cos tends to a constant at the poles
    /// and the polar easterlies become the strongest winds on the planet.
    fn psi(self, phi: f32) -> f32 {
        self.sign * self.amplitude * self.phase(phi).sin() * phi.cos()
    }

    fn dpsi_dphi(self, phi: f32) -> f32 {
        let t = self.phase(phi);
        self.sign
            * self.amplitude
            * (std::f32::consts::PI * t.cos() * phi.cos() / self.width() - t.sin() * phi.sin())
    }

    fn ascent_latitude(self) -> f32 {
        if self.sign > 0.0 {
            self.phi1
        } else {
            self.phi0
        }
    }

    /// Thermally direct cells (Hadley, polar) ascend at their equatorward
    /// edge; indirect (eddy-driven, Ferrel) cells ascend at the poleward one.
    fn is_direct(self) -> bool {
        let ascent = self.ascent_latitude();
        ascent.abs() <= self.phi0.abs().min(self.phi1.abs()) + 1e-6
    }
}

/// Latitude-only overturning circulation model.
#[derive(Debug, Clone)]
pub struct Circulation {
    cells: Vec<CirculationCell>,
    rotation_rate: f32,
}

/// Analytic circulation values at one latitude.
#[derive(Debug, Clone, Copy)]
pub struct CirculationSample {
    pub meridional_wind: f32,
    pub zonal_wind: f32,
    pub vertical_motion: f32,
    pub upper_meridional_wind: f32,
    pub upper_zonal_wind: f32,
}

impl Circulation {
    /// Build the default Earth-like circulation from `PLANET_ROTATION_RATE`.
    pub fn default_planet() -> Self {
        Self::from_rotation_rate(PLANET_ROTATION_RATE)
    }

    /// Build an Earth-like cell layout for a chosen relative rotation rate.
    pub fn from_rotation_rate(rotation_rate: f32) -> Self {
        let rotation_rate = rotation_rate.max(1e-6);
        let raw_hadley = 30.0_f32.to_radians() / rotation_rate.sqrt();
        let min_hadley = 8.0_f32.to_radians();
        let max_hadley = 70.0_f32.to_radians();

        let north_cells = if raw_hadley >= max_hadley {
            vec![CirculationCell {
                phi0: 0.0,
                phi1: std::f32::consts::FRAC_PI_2,
                amplitude: CIRC_HADLEY_AMPLITUDE,
                sign: -1.0,
            }]
        } else {
            let hadley = raw_hadley.clamp(min_hadley, max_hadley);
            let ferrel = (2.0 * hadley).min(80.0_f32.to_radians());
            build_northern_cells(hadley, ferrel)
        };

        let mut cells = Vec::with_capacity(north_cells.len() * 2);
        for cell in north_cells.iter().rev() {
            cells.push(CirculationCell {
                phi0: -cell.phi1,
                phi1: -cell.phi0,
                amplitude: cell.amplitude,
                sign: -cell.sign,
            });
        }
        cells.extend(north_cells);

        Self {
            cells,
            rotation_rate,
        }
    }

    pub fn sample_latitude(&self, phi: f32) -> CirculationSample {
        let phi = phi.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        let cos_phi = phi.cos().abs().max(1e-4);
        let cell = self.cell_at(phi);

        let dpsi = cell.dpsi_dphi(phi);
        let psi = cell.psi(phi);
        let meridional_wind = CIRC_MERIDIONAL_SCALE * psi / cos_phi;
        let omega = self.rotation_rate;
        let coriolis = 2.0 * omega * phi.sin();
        let zonal_wind = (coriolis / CIRC_FRICTION.max(1e-6)) * meridional_wind;
        let vertical_motion = -CIRC_MERIDIONAL_SCALE * dpsi / cos_phi;

        let upper_meridional_wind = -meridional_wind;
        // Angular-momentum conservation only holds in thermally direct cells
        // (Hadley, polar), where it yields the subtropical jet. The indirect
        // Ferrel cell is eddy-driven and roughly barotropic: upper flow is
        // amplified surface flow (westerly jet aloft, as observed).
        let upper_zonal_wind = if cell.is_direct() {
            // Clamp each cell's AM wind in proportion to its streamfunction
            // amplitude: a jet can't carry more momentum than its cell's
            // overturning sustains, so the vigorous Hadley cell's jet outruns
            // the polar one instead of both pinning at a global clamp.
            let max_amplitude = self
                .cells
                .iter()
                .map(|c| c.amplitude)
                .fold(0.0f32, f32::max)
                .max(1e-6);
            let clamp = UPPER_WIND_MAX_SPEED * cell.amplitude / max_amplitude;
            upper_zonal_wind(phi, cell.ascent_latitude(), self.rotation_rate, clamp)
        } else {
            UPPER_BAROTROPIC_FACTOR * zonal_wind
        };

        CirculationSample {
            meridional_wind,
            zonal_wind,
            vertical_motion,
            upper_meridional_wind,
            upper_zonal_wind,
        }
    }

    pub fn surface_wind_at(&self, pos: Vec3) -> Vec3 {
        let phi = latitude(pos);
        let sample = self.sample_latitude(phi);
        tangent_north(pos) * sample.meridional_wind + tangent_east(pos) * sample.zonal_wind
    }

    pub fn upper_wind_at(&self, pos: Vec3) -> Vec3 {
        let phi = latitude(pos);
        let sample = self.sample_latitude(phi);
        tangent_north(pos) * sample.upper_meridional_wind
            + tangent_east(pos) * sample.upper_zonal_wind
    }

    pub fn vertical_motion_at(&self, pos: Vec3) -> f32 {
        self.sample_latitude(latitude(pos)).vertical_motion
    }

    #[cfg(test)]
    fn psi_at(&self, phi: f32) -> f32 {
        let phi = phi.clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);
        self.cell_at(phi).psi(phi)
    }

    fn cell_at(&self, phi: f32) -> CirculationCell {
        self.cells
            .iter()
            .copied()
            .find(|cell| cell.contains(phi))
            .unwrap_or_else(|| {
                if phi < 0.0 {
                    self.cells[0]
                } else {
                    self.cells[self.cells.len() - 1]
                }
            })
    }
}

fn build_northern_cells(hadley: f32, ferrel: f32) -> Vec<CirculationCell> {
    let mut cells = Vec::with_capacity(3);
    let pole = std::f32::consts::FRAC_PI_2;
    const MIN_WIDTH: f32 = 1.0_f32.to_radians();

    cells.push(CirculationCell {
        phi0: 0.0,
        phi1: hadley,
        amplitude: CIRC_HADLEY_AMPLITUDE,
        sign: -1.0,
    });

    if ferrel - hadley > MIN_WIDTH {
        cells.push(CirculationCell {
            phi0: hadley,
            phi1: ferrel,
            amplitude: CIRC_FERREL_AMPLITUDE,
            sign: 1.0,
        });
    }

    if pole - ferrel > MIN_WIDTH {
        cells.push(CirculationCell {
            phi0: ferrel,
            phi1: pole,
            amplitude: CIRC_POLAR_AMPLITUDE,
            sign: -1.0,
        });
    }

    cells
}

fn latitude(pos: Vec3) -> f32 {
    pos.y.clamp(-1.0, 1.0).asin()
}

fn tangent_east(pos: Vec3) -> Vec3 {
    let east = Vec3::Y.cross(pos);
    let len = east.length();
    if len < 1e-6 {
        Vec3::X
    } else {
        east / len
    }
}

fn tangent_north(pos: Vec3) -> Vec3 {
    let north = Vec3::Y - pos * pos.y;
    let len = north.length();
    if len < 1e-6 {
        Vec3::Z
    } else {
        north / len
    }
}

fn upper_zonal_wind(phi: f32, ascent_lat: f32, rotation_rate: f32, clamp: f32) -> f32 {
    let cos_phi = phi.cos().abs().max(1e-4);
    let raw = OMEGA_SURFACE_SPEED * rotation_rate * (ascent_lat.cos().powi(2) - phi.cos().powi(2))
        / cos_phi;
    raw.clamp(-clamp, clamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integrate_sphere_weighted_w(circ: &Circulation) -> f32 {
        let steps = 200_000;
        let dphi = std::f32::consts::PI / steps as f32;
        let mut sum = 0.0;
        for i in 0..steps {
            let phi = -std::f32::consts::FRAC_PI_2 + (i as f32 + 0.5) * dphi;
            let w = circ.sample_latitude(phi).vertical_motion;
            sum += w * phi.cos() * dphi;
        }
        sum
    }

    #[test]
    fn psi_is_continuous_at_cell_edges() {
        let circ = Circulation::from_rotation_rate(1.0);
        for cell in &circ.cells {
            assert!(cell.psi(cell.phi0).abs() < 1e-5);
            assert!(cell.psi(cell.phi1).abs() < 1e-5);
        }
    }

    #[test]
    fn earth_like_sign_convention_has_equatorial_ascent() {
        let circ = Circulation::from_rotation_rate(1.0);
        let just_north = circ.sample_latitude(0.001);
        let just_south = circ.sample_latitude(-0.001);
        assert!(circ.psi_at(0.001) < 0.0);
        assert!(circ.psi_at(-0.001) > 0.0);
        assert!(just_north.meridional_wind < 0.0);
        assert!(just_south.meridional_wind > 0.0);
        assert!(circ.sample_latitude(0.0).vertical_motion > 0.0);
    }

    #[test]
    fn zonal_bands_emerge_from_cell_signs() {
        let circ = Circulation::from_rotation_rate(1.0);
        assert!(circ.sample_latitude(15.0_f32.to_radians()).zonal_wind < 0.0);
        assert!(circ.sample_latitude(45.0_f32.to_radians()).zonal_wind > 0.0);
        assert!(circ.sample_latitude(75.0_f32.to_radians()).zonal_wind < 0.0);
        assert!(circ.sample_latitude(-15.0_f32.to_radians()).zonal_wind < 0.0);
        assert!(circ.sample_latitude(-45.0_f32.to_radians()).zonal_wind > 0.0);
        assert!(circ.sample_latitude(-75.0_f32.to_radians()).zonal_wind < 0.0);
    }

    #[test]
    fn vertical_motion_integrates_to_zero_over_sphere() {
        let circ = Circulation::from_rotation_rate(1.0);
        assert!(integrate_sphere_weighted_w(&circ).abs() < 1e-4);
    }

    #[test]
    fn slow_rotator_degrades_to_one_cell_per_hemisphere() {
        let circ = Circulation::from_rotation_rate(0.1);
        assert_eq!(circ.cells.len(), 2);
        assert!(circ.sample_latitude(0.0).vertical_motion > 0.0);
        assert!(circ.sample_latitude(80.0_f32.to_radians()).vertical_motion < 0.0);
    }
}
