//! Canonical conversions between Hex3 model coordinates and physical display units.
//!
//! Elevation and crust-column thickness are not the same unit. Elevation has an
//! explicit physical interpretation (`ELEVATION_UNIT_KM`); crust thickness is a
//! reference-column coordinate converted to elevation through isostasy.

use super::{ELEVATION_UNIT_KM, PLANET_RADIUS_KM};

/// Radial displacement per elevation unit for a geometrically true-scale globe.
pub const PHYSICAL_RELIEF_SCALE: f32 = ELEVATION_UNIT_KM / PLANET_RADIUS_KM;

#[inline]
pub fn elevation_to_km(elevation: f32) -> f32 {
    elevation * ELEVATION_UNIT_KM
}

#[inline]
pub fn km_to_elevation(km: f32) -> f32 {
    km / ELEVATION_UNIT_KM
}

#[inline]
pub fn elevation_to_meters(elevation: f32) -> f32 {
    elevation_to_km(elevation) * 1_000.0
}

#[inline]
pub fn meters_to_elevation(meters: f32) -> f32 {
    km_to_elevation(meters / 1_000.0)
}

#[inline]
pub fn arc_radians_to_km(radians: f32) -> f32 {
    radians * PLANET_RADIUS_KM
}

#[inline]
pub fn km_to_arc_radians(km: f32) -> f32 {
    km / PLANET_RADIUS_KM
}

#[inline]
pub fn solid_angle_to_km2(steradians: f32) -> f32 {
    steradians * PLANET_RADIUS_KM * PLANET_RADIUS_KM
}

/// Convert the simulation's native slope coordinate (`Δelevation / Δradian`)
/// to physical surface grade (`vertical km / horizontal km = tan(angle)`).
#[inline]
pub fn elevation_per_radian_to_grade(slope: f32) -> f32 {
    slope * ELEVATION_UNIT_KM / PLANET_RADIUS_KM
}

#[inline]
pub fn grade_to_elevation_per_radian(grade: f32) -> f32 {
    grade * PLANET_RADIUS_KM / ELEVATION_UNIT_KM
}

#[inline]
pub fn grade_to_degrees(grade: f32) -> f32 {
    grade.atan().to_degrees()
}

#[inline]
pub fn relief_exaggeration(relief_scale: f32) -> f32 {
    relief_scale / PHYSICAL_RELIEF_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_conversions_round_trip() {
        let elevation = 0.73;
        assert!((km_to_elevation(elevation_to_km(elevation)) - elevation).abs() < 1e-6);
        assert!((meters_to_elevation(elevation_to_meters(elevation)) - elevation).abs() < 1e-6);

        let arc = 0.123;
        assert!((km_to_arc_radians(arc_radians_to_km(arc)) - arc).abs() < 1e-6);
    }

    #[test]
    fn physical_relief_is_exactly_one_x() {
        assert!((relief_exaggeration(PHYSICAL_RELIEF_SCALE) - 1.0).abs() < 1e-6);
        let elevation = km_to_elevation(8.0);
        let radial_km = elevation * PHYSICAL_RELIEF_SCALE * PLANET_RADIUS_KM;
        assert!((radial_km - 8.0).abs() < 1e-5);
    }

    #[test]
    fn simulation_slope_converts_to_physical_grade() {
        let grade = 0.5;
        let code_slope = grade_to_elevation_per_radian(grade);
        assert!((elevation_per_radian_to_grade(code_slope) - grade).abs() < 1e-6);
        assert!((grade_to_degrees(1.0) - 45.0).abs() < 1e-5);
    }
}
