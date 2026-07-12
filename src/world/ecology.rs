//! Seasonless ecological potentials derived from retained surface state.
//!
//! This is an interpretive semantic layer, not a dynamic ecosystem. It turns
//! normalized climate, physical terrain, and freshwater access into continuous
//! constraints first; categorical biome labels are deliberately secondary.

use serde::Serialize;

use super::diagnostics::distance_from_mask;
use super::{
    elevation_per_radian_to_grade, elevation_to_km, Hydrology, RiverSelection,
    RiverThresholdPolicy, Tessellation, PLANET_RADIUS_KM,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum BiomeKind {
    Ocean,
    Lake,
    Ice,
    Alpine,
    Tundra,
    BorealForest,
    TemperateForest,
    TropicalForest,
    Grassland,
    Shrubland,
    Steppe,
    Desert,
    Wetland,
    Barren,
}

impl BiomeKind {
    pub const LAND_KINDS: [Self; 12] = [
        Self::Ice,
        Self::Alpine,
        Self::Tundra,
        Self::BorealForest,
        Self::TemperateForest,
        Self::TropicalForest,
        Self::Grassland,
        Self::Shrubland,
        Self::Steppe,
        Self::Desert,
        Self::Wetland,
        Self::Barren,
    ];
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct EcologicalPotentials {
    /// Thermal opportunity on the model's normalized, seasonless temperature field.
    pub heat: f32,
    /// Relative moisture supply after a temperature-dependent demand proxy.
    pub moisture: f32,
    pub cold_stress: f32,
    pub water_stress: f32,
    pub alpine_stress: f32,
    pub terrain_stress: f32,
    pub freshwater_access: f32,
    pub vegetation: f32,
    pub tree: f32,
    pub wetland: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct EcologicalCell {
    pub biome: BiomeKind,
    /// Dominance of the winning label over the runner-up, in 0..1.
    /// Low values identify semantic transition zones.
    pub classification_confidence: f32,
    /// P / demand, normalized so area-weighted mean land aridity is one.
    pub aridity_index: f32,
    pub elevation_km: f32,
    pub physical_grade: f32,
    pub freshwater_distance_km: f32,
    pub potentials: EcologicalPotentials,
}

#[derive(Clone, Debug, Serialize)]
pub struct EcologySemantics {
    pub cells: Vec<EcologicalCell>,
    pub land_mean_raw_aridity: f32,
    /// True while temperature lacks seasonality and precipitation lacks a
    /// calibrated physical unit. Consumers must not present labels as Köppen.
    pub seasonless_proxy: bool,
}

impl EcologySemantics {
    pub fn build(
        tessellation: &Tessellation,
        elevation: &[f32],
        temperature: &[f32],
        precipitation: &[f32],
        hydrology: Option<&Hydrology>,
    ) -> Self {
        let n = tessellation.num_cells();
        assert_eq!(elevation.len(), n);
        assert_eq!(temperature.len(), n);
        assert_eq!(precipitation.len(), n);
        if let Some(hydrology) = hydrology {
            assert_eq!(hydrology.drainage_dir.len(), n);
        }

        let submerged: Vec<bool> = (0..n)
            .map(|cell| {
                hydrology
                    .map(|h| h.is_submerged(cell))
                    .unwrap_or(elevation[cell] < 0.0)
            })
            .collect();
        let raw_aridity: Vec<f32> = (0..n)
            .map(|cell| {
                let demand = 0.2 + 0.8 * temperature[cell].clamp(0.0, 1.0);
                precipitation[cell].max(0.0) / demand.max(1e-6)
            })
            .collect();
        let areas = tessellation.cell_areas();
        let (aridity_sum, land_area) =
            (0..n)
                .filter(|&cell| !submerged[cell])
                .fold((0.0f64, 0.0f64), |(sum, area), cell| {
                    (
                        sum + raw_aridity[cell] as f64 * areas[cell] as f64,
                        area + areas[cell] as f64,
                    )
                });
        let land_mean_raw_aridity = (aridity_sum / land_area.max(1e-30)) as f32;

        let freshwater = freshwater_mask(hydrology, n);
        let freshwater_distance = if freshwater.iter().any(|&cell| cell) {
            distance_from_mask(tessellation, &freshwater)
                .into_iter()
                .map(|radians| radians * PLANET_RADIUS_KM)
                .collect()
        } else {
            vec![f32::INFINITY; n]
        };

        let cells = (0..n)
            .map(|cell| {
                let elevation_km = elevation_to_km(elevation[cell]);
                let physical_grade = max_physical_grade(tessellation, elevation, cell);
                let aridity_index = raw_aridity[cell] / land_mean_raw_aridity.max(1e-6);
                let freshwater_distance_km = freshwater_distance[cell];
                let potentials = potentials(
                    temperature[cell],
                    aridity_index,
                    elevation_km,
                    physical_grade,
                    freshwater_distance_km,
                );

                let (biome, classification_confidence) = if submerged[cell] {
                    let ocean = hydrology.map(|h| h.is_ocean(cell)).unwrap_or(true);
                    (
                        if ocean {
                            BiomeKind::Ocean
                        } else {
                            BiomeKind::Lake
                        },
                        1.0,
                    )
                } else {
                    classify(temperature[cell], aridity_index, potentials)
                };

                EcologicalCell {
                    biome,
                    classification_confidence,
                    aridity_index,
                    elevation_km,
                    physical_grade,
                    freshwater_distance_km,
                    potentials,
                }
            })
            .collect();

        Self {
            cells,
            land_mean_raw_aridity,
            seasonless_proxy: true,
        }
    }
}

fn freshwater_mask(hydrology: Option<&Hydrology>, n: usize) -> Vec<bool> {
    let Some(hydrology) = hydrology else {
        return vec![false; n];
    };
    let rivers = RiverSelection::build(hydrology, RiverThresholdPolicy::default());
    (0..n)
        .map(|cell| {
            (hydrology.is_submerged(cell) && !hydrology.is_ocean(cell)) || rivers.all_cells[cell]
        })
        .collect()
}

fn max_physical_grade(tessellation: &Tessellation, elevation: &[f32], cell: usize) -> f32 {
    let center = tessellation.cell_center(cell);
    tessellation
        .neighbors(cell)
        .iter()
        .map(|&neighbor| {
            let radians = center
                .dot(tessellation.cell_center(neighbor))
                .clamp(-1.0, 1.0)
                .acos()
                .max(1e-8);
            elevation_per_radian_to_grade((elevation[cell] - elevation[neighbor]).abs() / radians)
        })
        .fold(0.0, f32::max)
}

fn potentials(
    temperature: f32,
    aridity: f32,
    elevation_km: f32,
    grade: f32,
    freshwater_distance_km: f32,
) -> EcologicalPotentials {
    let heat = smoothstep(-0.10, 0.75, temperature);
    let cold_stress = 1.0 - smoothstep(0.05, 0.32, temperature);
    let moisture = smoothstep(0.20, 1.60, aridity);
    let water_stress = 1.0 - smoothstep(0.25, 0.85, aridity);
    let alpine_stress = smoothstep(2.0, 4.5, elevation_km.max(0.0));
    let terrain_stress = smoothstep(0.20, 0.75, grade);
    let freshwater_access = if freshwater_distance_km.is_finite() {
        (-freshwater_distance_km / 35.0).exp()
    } else {
        0.0
    };
    let thermal_growth = smoothstep(-0.08, 0.20, temperature);
    let vegetation =
        (thermal_growth * moisture * (1.0 - 0.75 * alpine_stress) * (1.0 - 0.45 * terrain_stress))
            .clamp(0.0, 1.0);
    let tree = (vegetation
        * smoothstep(0.55, 1.35, aridity)
        * (1.0 - cold_stress)
        * (1.0 - alpine_stress))
        .clamp(0.0, 1.0);
    let wetland = (freshwater_access
        * smoothstep(0.65, 1.35, aridity)
        * (1.0 - terrain_stress)
        * thermal_growth)
        .clamp(0.0, 1.0);

    EcologicalPotentials {
        heat,
        moisture,
        cold_stress,
        water_stress,
        alpine_stress,
        terrain_stress,
        freshwater_access,
        vegetation,
        tree,
        wetland,
    }
}

fn classify(temperature: f32, aridity: f32, p: EcologicalPotentials) -> (BiomeKind, f32) {
    let warm = smoothstep(0.58, 0.82, temperature);
    let cool = triangular(temperature, 0.10, 0.34, 0.58);
    let temperate = triangular(temperature, 0.28, 0.55, 0.82);
    let dry_window = triangular(aridity, 0.20, 0.48, 0.90);
    let scores = [
        (BiomeKind::Ice, p.cold_stress * (1.0 - p.vegetation)),
        (
            BiomeKind::Alpine,
            p.alpine_stress * (0.5 + 0.5 * p.cold_stress),
        ),
        (
            BiomeKind::Tundra,
            p.cold_stress * (1.0 - p.alpine_stress) * 0.9,
        ),
        (BiomeKind::BorealForest, p.tree * cool),
        (BiomeKind::TemperateForest, p.tree * temperate),
        (BiomeKind::TropicalForest, p.tree * warm),
        (
            BiomeKind::Grassland,
            p.vegetation * (1.0 - p.tree) * smoothstep(0.45, 1.0, aridity),
        ),
        (
            BiomeKind::Shrubland,
            p.vegetation * (1.0 - p.tree) * dry_window,
        ),
        (BiomeKind::Steppe, dry_window * (0.45 + 0.55 * p.heat)),
        (BiomeKind::Desert, p.water_stress * (0.35 + 0.65 * p.heat)),
        (BiomeKind::Wetland, p.wetland),
        (
            BiomeKind::Barren,
            (1.0 - p.vegetation) * (0.3 + 0.7 * p.terrain_stress),
        ),
    ];
    let mut best = scores[0];
    let mut second = 0.0f32;
    for candidate in scores.into_iter().skip(1) {
        if candidate.1 > best.1 {
            second = best.1;
            best = candidate;
        } else {
            second = second.max(candidate.1);
        }
    }
    let confidence = ((best.1 - second) / best.1.max(1e-6)).clamp(0.0, 1.0);
    (best.0, confidence)
}

fn smoothstep(lo: f32, hi: f32, value: f32) -> f32 {
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn triangular(value: f32, lo: f32, peak: f32, hi: f32) -> f32 {
    if value <= peak {
        smoothstep(lo, peak, value)
    } else {
        1.0 - smoothstep(peak, hi, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn potentials_respond_monotonically_to_water_and_height() {
        let dry = potentials(0.7, 0.2, 0.2, 0.05, 500.0);
        let wet = potentials(0.7, 1.5, 0.2, 0.05, 5.0);
        assert!(wet.vegetation > dry.vegetation);
        assert!(wet.tree > dry.tree);
        assert!(wet.wetland > dry.wetland);

        let high = potentials(0.2, 1.5, 4.5, 0.05, 5.0);
        assert!(high.alpine_stress > wet.alpine_stress);
        assert!(high.vegetation < wet.vegetation);
    }

    #[test]
    fn categorical_labels_follow_dominant_constraints() {
        let desert = potentials(0.9, 0.1, 0.1, 0.02, 500.0);
        assert_eq!(classify(0.9, 0.1, desert).0, BiomeKind::Desert);

        let alpine = potentials(0.1, 1.0, 5.0, 0.3, 100.0);
        assert_eq!(classify(0.1, 1.0, alpine).0, BiomeKind::Alpine);
    }
}
