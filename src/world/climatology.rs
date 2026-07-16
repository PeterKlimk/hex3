//! Frozen-terrain climatology null for dossier-side causal comparison.
//!
//! This is intentionally an in-sample diagnostic, not an alternative climate
//! model. It asks how much of the retained precipitation and water geography is
//! reproduced by latitude, elevation, and distance from the authoritative
//! connected ocean alone.

use serde::Serialize;

use super::diagnostics::distance_from_mask;
use super::water_geography::RiverRoleSummary;
use super::{
    elevation_to_km, Elevation, Hydrology, RiverNetwork, RiverThresholdPolicy, Tessellation,
    WaterBodySemantics, WaterGeographyReport,
};

pub const CLIMATOLOGY_NULL_REPORT_SCHEMA_VERSION: u32 = 1;
const MARGINAL_BIN_COUNT: usize = 4;
const ESTIMATOR: &str = "land-only in-sample area-weighted joint conditional mean: signed-sin-latitude x pre-hydrology-elevation x connected-ocean-distance";

#[derive(Debug, Serialize)]
pub struct ClimatologyNullReport {
    pub schema_version: u32,
    pub estimator: &'static str,
    pub marginal_bin_count: usize,
    pub ocean_source_cell_count: usize,
    pub occupied_joint_bin_count: usize,
    pub smallest_occupied_land_area_fraction: f32,
    pub largest_occupied_land_area_fraction: f32,
    pub precipitation: PrecipitationNullComparison,
    pub frozen_terrain: FrozenTerrainComparison,
    pub baseline_water_geography: WaterGeographyReport,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PrecipitationNullComparison {
    pub product_land_mean: f32,
    pub null_land_mean: f32,
    pub product_land_stddev: f32,
    pub null_land_stddev: f32,
    pub residual_mae: f32,
    pub residual_rmse: f32,
    /// `1 - SSE / SST`; may be negative when the null is worse than the mean.
    pub explained_variance_fraction: f32,
    pub pearson_correlation: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct FrozenTerrainComparison {
    pub ocean_mask_disagreement_cell_count: usize,
    pub ocean_mask_disagreement_area_fraction: f32,
    pub drainage_direction_disagreement_cell_count: usize,
    pub drainage_direction_disagreement_land_area_fraction: f32,
    pub integration_cut_cell_disagreement_count: usize,
    pub basin_membership_disagreement_cell_count: usize,
    pub maximum_effective_elevation_difference_km: f32,
    pub flow_accumulation_land_correlation: f32,
    pub lake_water_cell_jaccard: f32,
    pub all_river_cell_jaccard: f32,
    pub major_river_cell_jaccard: f32,
    pub mouth_cell_jaccard: f32,
    pub highest_discharge_mouth_preserved: bool,
    pub longest_trunk_mouth_preserved: bool,
    pub highest_order_mouth_preserved: bool,
}

struct NullFit {
    precipitation: Vec<f32>,
    occupied_joint_bin_count: usize,
    smallest_occupied_land_area_fraction: f32,
    largest_occupied_land_area_fraction: f32,
}

impl ClimatologyNullReport {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        tessellation: &Tessellation,
        pre_hydrology_elevation: &Elevation,
        continentality: &[f32],
        temperature: &[f32],
        product_precipitation: &[f32],
        product_hydrology: &Hydrology,
        product_rivers: &RiverNetwork,
        product_water_geography: &WaterGeographyReport,
        river_policy: RiverThresholdPolicy,
    ) -> Result<Self, String> {
        let n = tessellation.num_cells();
        validate_inputs(
            n,
            pre_hydrology_elevation,
            continentality,
            temperature,
            product_precipitation,
            product_hydrology,
            product_rivers,
            product_water_geography,
        )?;

        let areas = tessellation.cell_areas_ref();
        let product_ocean: Vec<bool> = (0..n)
            .map(|cell| product_hydrology.is_ocean(cell))
            .collect();
        let ocean_source_cell_count = product_ocean.iter().filter(|&&ocean| ocean).count();
        let ocean_distance = if ocean_source_cell_count == 0 {
            vec![0.0; n]
        } else {
            distance_from_mask(tessellation, &product_ocean)
        };
        let signed_sin_latitude: Vec<f32> = (0..n)
            .map(|cell| tessellation.cell_center(cell).y)
            .collect();
        let fit = fit_conditional_null(
            &signed_sin_latitude,
            &pre_hydrology_elevation.values,
            &ocean_distance,
            product_precipitation,
            areas,
            &product_ocean,
        )?;
        let precipitation = precipitation_comparison(
            product_precipitation,
            &fit.precipitation,
            areas,
            &product_ocean,
        );

        let baseline_hydrology = Hydrology::generate_from_continentality(
            tessellation,
            continentality,
            pre_hydrology_elevation,
            &fit.precipitation,
            temperature,
        );
        let baseline_water = WaterBodySemantics::build(tessellation, &baseline_hydrology);
        let baseline_rivers = RiverNetwork::build(
            tessellation,
            &baseline_hydrology,
            &baseline_water,
            river_policy,
        );
        let baseline_water_geography = WaterGeographyReport::build(
            tessellation,
            &baseline_hydrology,
            &baseline_water,
            &baseline_rivers,
        )?;
        let frozen_terrain = compare_frozen_terrain(
            areas,
            product_hydrology,
            product_rivers,
            product_water_geography,
            &baseline_hydrology,
            &baseline_rivers,
            &baseline_water_geography,
        );

        Ok(Self {
            schema_version: CLIMATOLOGY_NULL_REPORT_SCHEMA_VERSION,
            estimator: ESTIMATOR,
            marginal_bin_count: MARGINAL_BIN_COUNT,
            ocean_source_cell_count,
            occupied_joint_bin_count: fit.occupied_joint_bin_count,
            smallest_occupied_land_area_fraction: fit.smallest_occupied_land_area_fraction,
            largest_occupied_land_area_fraction: fit.largest_occupied_land_area_fraction,
            precipitation,
            frozen_terrain,
            baseline_water_geography,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_inputs(
    n: usize,
    elevation: &Elevation,
    continentality: &[f32],
    temperature: &[f32],
    precipitation: &[f32],
    hydrology: &Hydrology,
    rivers: &RiverNetwork,
    water_geography: &WaterGeographyReport,
) -> Result<(), String> {
    if elevation.values.len() != n
        || continentality.len() != n
        || temperature.len() != n
        || precipitation.len() != n
        || hydrology.elevation.len() != n
        || hydrology.is_ocean.len() != n
        || hydrology.drainage_dir.len() != n
        || hydrology.flow_accumulation.len() != n
        || rivers.all_cells.len() != n
        || rivers.major_cells.len() != n
        || rivers.upstream.len() != n
        || rivers.strahler_order.len() != n
        || water_geography.cell_count != n
    {
        return Err("climatology null input/tessellation length mismatch".into());
    }
    if precipitation
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || elevation.values.iter().any(|value| !value.is_finite())
        || continentality.iter().any(|value| !value.is_finite())
        || temperature.iter().any(|value| !value.is_finite())
    {
        return Err("climatology null requires finite inputs and nonnegative precipitation".into());
    }
    if hydrology.is_ocean.iter().all(|&ocean| ocean) {
        return Err("climatology null requires at least one land cell".into());
    }
    Ok(())
}

fn fit_conditional_null(
    latitude: &[f32],
    elevation: &[f32],
    ocean_distance: &[f32],
    precipitation: &[f32],
    areas: &[f32],
    ocean: &[bool],
) -> Result<NullFit, String> {
    let n = precipitation.len();
    if latitude.len() != n
        || elevation.len() != n
        || ocean_distance.len() != n
        || areas.len() != n
        || ocean.len() != n
    {
        return Err("climatology estimator input length mismatch".into());
    }
    let land: Vec<bool> = ocean.iter().map(|&value| !value).collect();
    let land_area: f64 = (0..n)
        .filter(|&cell| land[cell])
        .map(|cell| areas[cell] as f64)
        .sum();
    if land_area <= 0.0 {
        return Err("climatology null requires positive land area".into());
    }

    let lat_bin = marginal_quantile_bins(latitude, areas, &land);
    let elevation_bin = marginal_quantile_bins(elevation, areas, &land);
    let distance_bin = if ocean.iter().any(|&value| value) {
        marginal_quantile_bins(ocean_distance, areas, &land)
    } else {
        vec![0; n]
    };
    let joint_count = MARGINAL_BIN_COUNT.pow(3);
    let mut bin_area = vec![0.0f64; joint_count];
    let mut bin_supply = vec![0.0f64; joint_count];
    for cell in 0..n {
        if land[cell] {
            let bin = joint_bin(lat_bin[cell], elevation_bin[cell], distance_bin[cell]);
            let area = areas[cell] as f64;
            bin_area[bin] += area;
            bin_supply[bin] += area * precipitation[cell] as f64;
        }
    }
    let mut prediction = precipitation.to_vec();
    for cell in 0..n {
        if land[cell] {
            let bin = joint_bin(lat_bin[cell], elevation_bin[cell], distance_bin[cell]);
            prediction[cell] = (bin_supply[bin] / bin_area[bin]) as f32;
        }
    }

    // Casting bin means to f32 introduces a tiny budget error. Remove it with
    // one global scale so the hydrology comparison holds land supply fixed.
    let target_supply = weighted_sum(precipitation, areas, &land);
    let predicted_supply = weighted_sum(&prediction, areas, &land);
    if predicted_supply > 0.0 {
        let scale = (target_supply / predicted_supply) as f32;
        for cell in 0..n {
            if land[cell] {
                prediction[cell] *= scale;
            }
        }
    } else if target_supply > 0.0 {
        return Err("climatology null produced zero supply from positive precipitation".into());
    }

    let occupied: Vec<f64> = bin_area.into_iter().filter(|area| *area > 0.0).collect();
    let smallest = occupied.iter().copied().fold(f64::INFINITY, f64::min) / land_area;
    let largest = occupied.iter().copied().fold(0.0f64, f64::max) / land_area;
    Ok(NullFit {
        precipitation: prediction,
        occupied_joint_bin_count: occupied.len(),
        smallest_occupied_land_area_fraction: smallest as f32,
        largest_occupied_land_area_fraction: largest as f32,
    })
}

fn marginal_quantile_bins(values: &[f32], areas: &[f32], land: &[bool]) -> Vec<usize> {
    let mut ordered: Vec<usize> = (0..values.len()).filter(|&cell| land[cell]).collect();
    ordered.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| a.cmp(&b)));
    let total: f64 = ordered.iter().map(|&cell| areas[cell] as f64).sum();
    let mut thresholds = [0.0f32; MARGINAL_BIN_COUNT - 1];
    let mut cumulative = 0.0f64;
    let mut threshold_index = 0;
    for &cell in &ordered {
        cumulative += areas[cell] as f64;
        while threshold_index < thresholds.len()
            && cumulative >= total * (threshold_index + 1) as f64 / MARGINAL_BIN_COUNT as f64
        {
            thresholds[threshold_index] = values[cell];
            threshold_index += 1;
        }
    }
    (0..values.len())
        .map(|cell| {
            thresholds
                .iter()
                .position(|&threshold| values[cell] <= threshold)
                .unwrap_or(MARGINAL_BIN_COUNT - 1)
        })
        .collect()
}

#[inline]
fn joint_bin(latitude: usize, elevation: usize, distance: usize) -> usize {
    (latitude * MARGINAL_BIN_COUNT + elevation) * MARGINAL_BIN_COUNT + distance
}

fn precipitation_comparison(
    product: &[f32],
    baseline: &[f32],
    areas: &[f32],
    ocean: &[bool],
) -> PrecipitationNullComparison {
    let land: Vec<bool> = ocean.iter().map(|&value| !value).collect();
    let (product_mean, product_variance) = weighted_mean_variance(product, areas, &land);
    let (baseline_mean, baseline_variance) = weighted_mean_variance(baseline, areas, &land);
    let area: f64 = (0..product.len())
        .filter(|&cell| land[cell])
        .map(|cell| areas[cell] as f64)
        .sum();
    let mut absolute_error = 0.0f64;
    let mut squared_error = 0.0f64;
    for cell in 0..product.len() {
        if land[cell] {
            let residual = product[cell] as f64 - baseline[cell] as f64;
            absolute_error += areas[cell] as f64 * residual.abs();
            squared_error += areas[cell] as f64 * residual * residual;
        }
    }
    let explained = if product_variance <= f64::EPSILON {
        if squared_error <= f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - squared_error / (area * product_variance)
    };
    PrecipitationNullComparison {
        product_land_mean: product_mean as f32,
        null_land_mean: baseline_mean as f32,
        product_land_stddev: product_variance.sqrt() as f32,
        null_land_stddev: baseline_variance.sqrt() as f32,
        residual_mae: (absolute_error / area) as f32,
        residual_rmse: (squared_error / area).sqrt() as f32,
        explained_variance_fraction: explained as f32,
        pearson_correlation: weighted_correlation(product, baseline, areas, &land),
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_frozen_terrain(
    areas: &[f32],
    product: &Hydrology,
    product_rivers: &RiverNetwork,
    product_report: &WaterGeographyReport,
    baseline: &Hydrology,
    baseline_rivers: &RiverNetwork,
    baseline_report: &WaterGeographyReport,
) -> FrozenTerrainComparison {
    let n = areas.len();
    let total_area: f64 = areas.iter().map(|&area| area as f64).sum();
    let land: Vec<bool> = (0..n).map(|cell| !product.is_ocean(cell)).collect();
    let land_area: f64 = (0..n)
        .filter(|&cell| land[cell])
        .map(|cell| areas[cell] as f64)
        .sum();
    let ocean_disagreement: Vec<bool> = (0..n)
        .map(|cell| product.is_ocean(cell) != baseline.is_ocean(cell))
        .collect();
    let drainage_disagreement: Vec<bool> = (0..n)
        .map(|cell| land[cell] && product.downstream(cell) != baseline.downstream(cell))
        .collect();
    let integration_cut_disagreement: Vec<bool> = (0..n)
        .map(|cell| {
            product.was_lowered_by_integration(cell) != baseline.was_lowered_by_integration(cell)
        })
        .collect();

    FrozenTerrainComparison {
        ocean_mask_disagreement_cell_count: ocean_disagreement.iter().filter(|&&v| v).count(),
        ocean_mask_disagreement_area_fraction: mask_area(areas, &ocean_disagreement)
            / total_area as f32,
        drainage_direction_disagreement_cell_count: drainage_disagreement
            .iter()
            .filter(|&&v| v)
            .count(),
        drainage_direction_disagreement_land_area_fraction: if land_area > 0.0 {
            mask_area(areas, &drainage_disagreement) / land_area as f32
        } else {
            0.0
        },
        integration_cut_cell_disagreement_count: integration_cut_disagreement
            .iter()
            .filter(|&&value| value)
            .count(),
        basin_membership_disagreement_cell_count: (0..n)
            .filter(|&cell| product.basin_id[cell] != baseline.basin_id[cell])
            .count(),
        maximum_effective_elevation_difference_km: (0..n)
            .map(|cell| elevation_to_km((product.elevation[cell] - baseline.elevation[cell]).abs()))
            .fold(0.0, f32::max),
        flow_accumulation_land_correlation: weighted_correlation(
            &product.flow_accumulation,
            &baseline.flow_accumulation,
            areas,
            &land,
        ),
        lake_water_cell_jaccard: predicate_jaccard(
            n,
            |cell| product.is_lake_water(cell),
            |cell| baseline.is_lake_water(cell),
        ),
        all_river_cell_jaccard: boolean_jaccard(
            &product_rivers.all_cells,
            &baseline_rivers.all_cells,
        ),
        major_river_cell_jaccard: boolean_jaccard(
            &product_rivers.major_cells,
            &baseline_rivers.major_cells,
        ),
        mouth_cell_jaccard: set_jaccard(n, &product_rivers.mouths, &baseline_rivers.mouths),
        highest_discharge_mouth_preserved: same_role_mouth(
            &product_report.rivers.highest_discharge,
            &baseline_report.rivers.highest_discharge,
        ),
        longest_trunk_mouth_preserved: same_role_mouth(
            &product_report.rivers.longest_trunk,
            &baseline_report.rivers.longest_trunk,
        ),
        highest_order_mouth_preserved: same_role_mouth(
            &product_report.rivers.highest_order,
            &baseline_report.rivers.highest_order,
        ),
    }
}

fn weighted_sum(values: &[f32], areas: &[f32], mask: &[bool]) -> f64 {
    (0..values.len())
        .filter(|&cell| mask[cell])
        .map(|cell| values[cell] as f64 * areas[cell] as f64)
        .sum()
}

fn weighted_mean_variance(values: &[f32], areas: &[f32], mask: &[bool]) -> (f64, f64) {
    let area: f64 = (0..values.len())
        .filter(|&cell| mask[cell])
        .map(|cell| areas[cell] as f64)
        .sum();
    let mean = weighted_sum(values, areas, mask) / area;
    let variance = (0..values.len())
        .filter(|&cell| mask[cell])
        .map(|cell| {
            let delta = values[cell] as f64 - mean;
            areas[cell] as f64 * delta * delta
        })
        .sum::<f64>()
        / area;
    (mean, variance.max(0.0))
}

fn weighted_correlation(a: &[f32], b: &[f32], areas: &[f32], mask: &[bool]) -> f32 {
    let (mean_a, variance_a) = weighted_mean_variance(a, areas, mask);
    let (mean_b, variance_b) = weighted_mean_variance(b, areas, mask);
    if variance_a <= f64::EPSILON && variance_b <= f64::EPSILON {
        return if (mean_a - mean_b).abs() <= 1e-12 {
            1.0
        } else {
            0.0
        };
    }
    if variance_a <= f64::EPSILON || variance_b <= f64::EPSILON {
        return 0.0;
    }
    let area: f64 = (0..a.len())
        .filter(|&cell| mask[cell])
        .map(|cell| areas[cell] as f64)
        .sum();
    let covariance: f64 = (0..a.len())
        .filter(|&cell| mask[cell])
        .map(|cell| areas[cell] as f64 * (a[cell] as f64 - mean_a) * (b[cell] as f64 - mean_b))
        .sum::<f64>()
        / area;
    (covariance / (variance_a * variance_b).sqrt()).clamp(-1.0, 1.0) as f32
}

fn mask_area(areas: &[f32], mask: &[bool]) -> f32 {
    let area: f32 = (0..areas.len())
        .filter(|&cell| mask[cell])
        .map(|cell| areas[cell])
        .sum();
    if area == 0.0 {
        0.0
    } else {
        area
    }
}

fn boolean_jaccard(a: &[bool], b: &[bool]) -> f32 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (&a, &b) in a.iter().zip(b) {
        intersection += usize::from(a && b);
        union += usize::from(a || b);
    }
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

fn predicate_jaccard(n: usize, a: impl Fn(usize) -> bool, b: impl Fn(usize) -> bool) -> f32 {
    let left: Vec<bool> = (0..n).map(a).collect();
    let right: Vec<bool> = (0..n).map(b).collect();
    boolean_jaccard(&left, &right)
}

fn set_jaccard(n: usize, a: &[usize], b: &[usize]) -> f32 {
    let mut left = vec![false; n];
    let mut right = vec![false; n];
    for &cell in a {
        left[cell] = true;
    }
    for &cell in b {
        right[cell] = true;
    }
    boolean_jaccard(&left, &right)
}

fn same_role_mouth(a: &Option<RiverRoleSummary>, b: &Option<RiverRoleSummary>) -> bool {
    a.as_ref().map(|role| role.mouth_cell) == b.as_ref().map(|role| role.mouth_cell)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::world::NoiseLayerData;

    fn tessellation(cells: usize) -> Tessellation {
        let mut rng = ChaCha8Rng::seed_from_u64(991);
        Tessellation::generate(cells, 0, &mut rng)
    }

    #[test]
    fn estimator_is_deterministic_nonnegative_and_preserves_land_supply() {
        let tess = tessellation(300);
        let n = tess.num_cells();
        let areas = tess.cell_areas_ref();
        let latitude: Vec<f32> = (0..n).map(|i| tess.cell_center(i).y).collect();
        let elevation: Vec<f32> = (0..n).map(|i| 0.2 * tess.cell_center(i).z).collect();
        let ocean: Vec<bool> = elevation.iter().map(|&z| z < -0.05).collect();
        let distance = if ocean.iter().any(|&v| v) {
            distance_from_mask(&tess, &ocean)
        } else {
            vec![0.0; n]
        };
        let precipitation: Vec<f32> = (0..n)
            .map(|i| (1.0 + 0.4 * tess.cell_center(i).x).max(0.0))
            .collect();
        let a = fit_conditional_null(
            &latitude,
            &elevation,
            &distance,
            &precipitation,
            areas,
            &ocean,
        )
        .unwrap();
        let b = fit_conditional_null(
            &latitude,
            &elevation,
            &distance,
            &precipitation,
            areas,
            &ocean,
        )
        .unwrap();
        assert_eq!(a.precipitation, b.precipitation);
        assert!(a
            .precipitation
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
        let land: Vec<bool> = ocean.iter().map(|&v| !v).collect();
        let expected = weighted_sum(&precipitation, areas, &land);
        let actual = weighted_sum(&a.precipitation, areas, &land);
        assert!((expected - actual).abs() <= expected.abs().max(1.0) * 2e-7);
    }

    #[test]
    fn refitting_own_conditional_means_has_negligible_residual() {
        let tess = tessellation(320);
        let n = tess.num_cells();
        let areas = tess.cell_areas_ref();
        let latitude: Vec<f32> = (0..n).map(|i| tess.cell_center(i).y).collect();
        let elevation: Vec<f32> = (0..n).map(|i| 0.3 * tess.cell_center(i).z).collect();
        let ocean: Vec<bool> = elevation.iter().map(|&z| z < -0.1).collect();
        let distance = distance_from_mask(&tess, &ocean);
        let initial: Vec<f32> = (0..n).map(|i| 1.0 + 0.3 * tess.cell_center(i).x).collect();
        let first = fit_conditional_null(&latitude, &elevation, &distance, &initial, areas, &ocean)
            .unwrap();
        let second = fit_conditional_null(
            &latitude,
            &elevation,
            &distance,
            &first.precipitation,
            areas,
            &ocean,
        )
        .unwrap();
        let comparison =
            precipitation_comparison(&first.precipitation, &second.precipitation, areas, &ocean);
        assert!(comparison.residual_rmse < 1e-6);
    }

    #[test]
    fn longitude_anomaly_is_not_absorbed_by_the_null() {
        let tess = tessellation(400);
        let n = tess.num_cells();
        let areas = tess.cell_areas_ref();
        let latitude: Vec<f32> = (0..n).map(|i| tess.cell_center(i).y).collect();
        let elevation = vec![0.1; n];
        let distance = vec![0.0; n];
        let ocean = vec![false; n];
        let precipitation: Vec<f32> = (0..n).map(|i| 1.0 + 0.5 * tess.cell_center(i).x).collect();
        let fit = fit_conditional_null(
            &latitude,
            &elevation,
            &distance,
            &precipitation,
            areas,
            &ocean,
        )
        .unwrap();
        let comparison =
            precipitation_comparison(&precipitation, &fit.precipitation, areas, &ocean);
        assert!(comparison.residual_rmse > 0.2);
    }

    #[test]
    fn paired_hydrology_report_smoke() {
        let tess = tessellation(240);
        let n = tess.num_cells();
        let elevation = Elevation {
            values: (0..n).map(|i| 0.2 * tess.cell_center(i).y).collect(),
            noise_contribution: vec![0.0; n],
            noise_layers: NoiseLayerData {
                macro_layer: vec![0.0; n],
            },
        };
        let continentality = vec![0.0; n];
        let temperature = vec![0.5; n];
        let precipitation: Vec<f32> = (0..n).map(|i| 1.0 + 0.2 * tess.cell_center(i).x).collect();
        let product = Hydrology::generate_from_continentality(
            &tess,
            &continentality,
            &elevation,
            &precipitation,
            &temperature,
        );
        let water = WaterBodySemantics::build(&tess, &product);
        let policy = RiverThresholdPolicy::default();
        let rivers = RiverNetwork::build(&tess, &product, &water, policy);
        let product_report = WaterGeographyReport::build(&tess, &product, &water, &rivers).unwrap();
        let report = ClimatologyNullReport::build(
            &tess,
            &elevation,
            &continentality,
            &temperature,
            &precipitation,
            &product,
            &rivers,
            &product_report,
            policy,
        )
        .unwrap();
        assert_eq!(report.baseline_water_geography.cell_count, n);
        assert_eq!(report.frozen_terrain.ocean_mask_disagreement_cell_count, 0);
        assert_eq!(
            report
                .frozen_terrain
                .drainage_direction_disagreement_cell_count,
            0
        );
        assert_eq!(
            report
                .frozen_terrain
                .integration_cut_cell_disagreement_count,
            0
        );
        assert_eq!(
            report
                .frozen_terrain
                .basin_membership_disagreement_cell_count,
            0
        );
        assert_eq!(
            report
                .frozen_terrain
                .maximum_effective_elevation_difference_km,
            0.0
        );
    }
}
