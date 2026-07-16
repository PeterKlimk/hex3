//! Cause-first equilibrium physiognomy derived from the retained surface.
//!
//! This is a bounded semantic product, not a dynamic vegetation model. It
//! preserves the dependency order `climate + hydrologic position -> opportunity
//! -> fractional cover` and deliberately does not assign biomes. A terrain-
//! exposure term waits for a scale-declared robust measure.

use std::collections::VecDeque;

use serde::Serialize;

use super::{elevation_to_km, solid_angle_to_km2, Hydrology, Tessellation};

/// Nominal physical catchment scale at which a drainage cell can anchor HAND.
///
/// This is independent of the cartographic river-selection policy. Two thousand
/// square kilometres is a world/regional semantic scale, not a claim that
/// smaller real streams do not exist. A four-local-cell numerical floor keeps
/// the reference network resolved on an adaptive mesh.
pub const LIVING_SURFACE_DRAINAGE_REFERENCE_KM2: f32 = 2_000.0;

const MIN_DRAINAGE_REFERENCE_CELLS: f32 = 4.0;
/// Floodplain-scale vertical decay length for drainage-relative saturation (30 m).
const HAND_SATURATION_SCALE_KM: f32 = 0.03;
/// HAND is sampled once per non-reference cell, but the result is a subcell
/// cover fraction.
const MAX_DRAINAGE_SATURATED_FRACTION: f32 = 0.35;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PhysiognomyFractions {
    pub bare: f32,
    pub herbaceous: f32,
    pub woody: f32,
    pub wetland: f32,
}

impl PhysiognomyFractions {
    pub fn terrestrial_sum(self) -> f32 {
        self.bare + self.herbaceous + self.woody + self.wetland
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct LivingSurfaceCell {
    /// Opportunity supplied by the normalized, seasonless temperature field.
    pub thermal_opportunity: f32,
    /// One is maximally water limited; zero is not water limited by climate.
    pub relative_water_limitation: f32,
    /// Drainage-relative wetness opportunity, not groundwater or soil moisture.
    pub drainage_saturation: f32,
    /// Height above the first downstream drainage/lake/ocean reference. `None`
    /// means submerged or terminating in an unresolved dry basin.
    pub height_above_drainage_km: Option<f32>,
    /// Combined equilibrium growth opportunity before conversion to cover.
    pub growth_opportunity: f32,
    /// Total terrestrial vegetation cover opportunity.
    pub vegetation_cover: f32,
    /// Woody share of non-wet vegetation opportunity.
    pub woody_share: f32,
    /// Exclusive semantic cover fractions. All are zero under water.
    pub fractions: PhysiognomyFractions,
}

#[derive(Clone, Debug, Serialize)]
pub struct LivingSurfaceSemantics {
    pub cells: Vec<LivingSurfaceCell>,
    /// Multiplier applied to the supplied final precipitation field. This is
    /// fixed at 1.0 until a planetary wetness control rebuilds both precipitation
    /// and hydrology from the same upstream setting; hydrology's lake-only
    /// `climate_ratio` is not that control.
    pub planetary_water_supply_scale: f32,
    /// Nominal physical threshold; each cell also requires at least
    /// `minimum_drainage_reference_cells` times its own area upstream.
    pub drainage_reference_area_km2: f32,
    pub minimum_drainage_reference_cells: f32,
    /// True until temperature gains seasons and precipitation/demand gain
    /// calibrated physical units.
    pub seasonless_relative_supply: bool,
}

impl LivingSurfaceSemantics {
    pub fn build(
        tessellation: &Tessellation,
        temperature: &[f32],
        precipitation: &[f32],
        hydrology: &Hydrology,
    ) -> Self {
        let n = tessellation.num_cells();
        assert_eq!(temperature.len(), n, "living-surface temperature length");
        assert_eq!(
            precipitation.len(),
            n,
            "living-surface precipitation length"
        );
        assert_eq!(
            hydrology.drainage_dir.len(),
            n,
            "living-surface hydrology length"
        );
        assert_eq!(
            hydrology.elevation.len(),
            n,
            "living-surface hydrologic elevation length"
        );
        assert!(
            temperature.iter().all(|value| value.is_finite()),
            "living-surface temperature must be finite"
        );
        assert!(
            precipitation
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "living-surface precipitation must be finite and non-negative"
        );
        assert!(
            hydrology.elevation.iter().all(|value| value.is_finite()),
            "living-surface hydrologic elevation must be finite"
        );
        let submerged: Vec<bool> = (0..n).map(|cell| hydrology.is_submerged(cell)).collect();
        let drainage = drainage_position(tessellation, hydrology, &submerged);

        let cells = (0..n)
            .map(|cell| {
                cover_response(
                    temperature[cell],
                    precipitation[cell],
                    1.0,
                    drainage[cell].saturation,
                    drainage[cell].hand_km,
                    submerged[cell],
                )
            })
            .collect();

        Self {
            cells,
            planetary_water_supply_scale: 1.0,
            drainage_reference_area_km2: LIVING_SURFACE_DRAINAGE_REFERENCE_KM2,
            minimum_drainage_reference_cells: MIN_DRAINAGE_REFERENCE_CELLS,
            seasonless_relative_supply: true,
        }
    }
}

fn cover_response(
    temperature: f32,
    precipitation: f32,
    planetary_water_supply_scale: f32,
    drainage_saturation: f32,
    height_above_drainage_km: Option<f32>,
    submerged: bool,
) -> LivingSurfaceCell {
    let temperature = finite_or_zero(temperature);
    let precipitation = finite_or_zero(precipitation).max(0.0);
    let drainage_saturation = finite_or_zero(drainage_saturation).clamp(0.0, 1.0);

    if submerged {
        return LivingSurfaceCell {
            thermal_opportunity: 0.0,
            relative_water_limitation: 0.0,
            drainage_saturation: 0.0,
            height_above_drainage_km: None,
            growth_opportunity: 0.0,
            vegetation_cover: 0.0,
            woody_share: 0.0,
            fractions: PhysiognomyFractions::default(),
        };
    }

    let thermal_opportunity = smoothstep(-0.05, 0.30, temperature);
    let demand = 0.25 + 0.75 * temperature.clamp(0.0, 1.0);
    let supply = precipitation * planetary_water_supply_scale;
    // A dimensionless saturating supply/demand relationship. This is not PET or
    // a root-zone water balance, so retain the "relative" name.
    let climatic_water_availability = supply / (supply + 0.70 * demand).max(1e-8);
    let relative_water_limitation = 1.0 - climatic_water_availability;

    // Drainage wetness can relieve climatic limitation locally, but cannot
    // create thermal opportunity.
    let drainage_water = drainage_saturation * planetary_water_supply_scale.clamp(0.0, 1.0);
    let effective_water = 1.0 - (1.0 - climatic_water_availability) * (1.0 - 0.75 * drainage_water);
    let growth_opportunity = (thermal_opportunity * effective_water).clamp(0.0, 1.0);
    let vegetation_cover = smoothstep(0.08, 0.72, growth_opportunity);
    let woody_share = (smoothstep(0.42, 0.72, effective_water)
        * smoothstep(0.12, 0.45, temperature))
    .clamp(0.0, 1.0);

    let fractions = partition_cover(vegetation_cover, woody_share, drainage_saturation);

    LivingSurfaceCell {
        thermal_opportunity,
        relative_water_limitation,
        drainage_saturation,
        height_above_drainage_km,
        growth_opportunity,
        vegetation_cover,
        woody_share,
        fractions,
    }
}

fn partition_cover(
    vegetation_cover: f32,
    woody_share: f32,
    drainage_saturation: f32,
) -> PhysiognomyFractions {
    let cover = vegetation_cover.clamp(0.0, 1.0);
    let wetland = cover * drainage_saturation.clamp(0.0, 1.0);
    let non_wet = cover - wetland;
    let woody = non_wet * woody_share.clamp(0.0, 1.0);
    let bare = 1.0 - cover;
    // Compute the residual last so the semantic partition closes despite
    // floating-point evaluation order.
    let herbaceous = (1.0 - bare - wetland - woody).max(0.0);
    PhysiognomyFractions {
        bare,
        herbaceous,
        woody,
        wetland,
    }
}

#[derive(Clone, Copy, Debug)]
struct DrainagePosition {
    hand_km: Option<f32>,
    saturation: f32,
}

fn drainage_position(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    submerged: &[bool],
) -> Vec<DrainagePosition> {
    let drainage_dir = &hydrology.drainage_dir;
    let area_km2: Vec<f64> = tessellation
        .cell_areas_ref()
        .iter()
        .map(|&steradians| solid_angle_to_km2(steradians) as f64)
        .collect();
    let (contributing_area, sources_first) = geometric_contributing_area(drainage_dir, &area_km2);

    let mut lake_surface_by_basin = vec![None; hydrology.basins.len()];
    for body in &hydrology.water_bodies {
        if body.is_lake {
            lake_surface_by_basin[body.basin_id] =
                Some(elevation_to_km(hydrology.basins[body.basin_id].water_level));
        }
    }
    let mut explicit_reference_km = vec![None; drainage_dir.len()];
    let mut channel_reference = vec![false; drainage_dir.len()];
    let mut terminal_dry_basin = vec![false; drainage_dir.len()];
    for cell in 0..drainage_dir.len() {
        if hydrology.is_ocean(cell) {
            explicit_reference_km[cell] = Some(0.0);
            continue;
        }
        if let Some(basin_id) = hydrology.basin_id[cell] {
            if let Some(surface_km) = lake_surface_by_basin[basin_id] {
                // All cells in the wet depression, including dry fringe, use
                // the lake surface rather than the priority-filled route.
                explicit_reference_km[cell] = Some(surface_km);
                continue;
            }
            if !hydrology.basins[basin_id].is_overflowing() {
                terminal_dry_basin[cell] = true;
                continue;
            }
        }
        if qualifies_as_drainage_reference(contributing_area[cell], area_km2[cell]) {
            explicit_reference_km[cell] = Some(elevation_to_km(hydrology.elevation[cell]));
            channel_reference[cell] = true;
        }
    }
    let reference_elevation_km = downstream_reference_elevations(
        drainage_dir,
        &explicit_reference_km,
        &terminal_dry_basin,
        &sources_first,
    );

    (0..drainage_dir.len())
        .map(|cell| {
            if submerged[cell] {
                return DrainagePosition {
                    hand_km: None,
                    saturation: 0.0,
                };
            }
            let Some(reference_elevation_km) = reference_elevation_km[cell] else {
                return DrainagePosition {
                    hand_km: None,
                    saturation: 0.0,
                };
            };
            let hand_km =
                (elevation_to_km(hydrology.elevation[cell]) - reference_elevation_km).max(0.0);
            DrainagePosition {
                hand_km: Some(hand_km),
                // Reference-network membership establishes a downstream datum;
                // it does not tell us how much of this cell is floodplain or
                // channel. Until a scale-declared width/valley-floor owner
                // exists, do not turn the whole reference corridor into wetland.
                saturation: saturation_from_hand(hand_km, channel_reference[cell]),
            }
        })
        .collect()
}

fn qualifies_as_drainage_reference(contributing_area_km2: f64, local_area_km2: f64) -> bool {
    contributing_area_km2
        >= f64::from(LIVING_SURFACE_DRAINAGE_REFERENCE_KM2)
            .max(f64::from(MIN_DRAINAGE_REFERENCE_CELLS) * local_area_km2)
}

fn saturation_from_hand(hand_km: f32, owns_channel_reference: bool) -> f32 {
    if owns_channel_reference {
        return 0.0;
    }
    (MAX_DRAINAGE_SATURATED_FRACTION * (-hand_km.max(0.0) / HAND_SATURATION_SCALE_KM).exp())
        .clamp(0.0, MAX_DRAINAGE_SATURATED_FRACTION)
}

fn geometric_contributing_area(
    drainage_dir: &[Option<usize>],
    local_area_km2: &[f64],
) -> (Vec<f64>, Vec<usize>) {
    let n = drainage_dir.len();
    assert_eq!(local_area_km2.len(), n);
    let mut upstream_count = vec![0usize; n];
    for (cell, downstream) in drainage_dir.iter().copied().enumerate() {
        if let Some(downstream) = downstream {
            assert!(downstream < n, "drainage receiver out of bounds at {cell}");
            assert_ne!(downstream, cell, "self drainage at cell {cell}");
            upstream_count[downstream] += 1;
        }
    }
    let mut remaining = upstream_count.clone();
    let mut ready: VecDeque<usize> = (0..n).filter(|&cell| remaining[cell] == 0).collect();
    let mut accumulated = local_area_km2.to_vec();
    let mut sources_first = Vec::with_capacity(n);
    while let Some(cell) = ready.pop_front() {
        sources_first.push(cell);
        if let Some(downstream) = drainage_dir[cell] {
            accumulated[downstream] += accumulated[cell];
            remaining[downstream] -= 1;
            if remaining[downstream] == 0 {
                ready.push_back(downstream);
            }
        }
    }
    assert_eq!(
        sources_first.len(),
        n,
        "living-surface drainage graph must be acyclic"
    );
    (accumulated, sources_first)
}

fn downstream_reference_elevations(
    drainage_dir: &[Option<usize>],
    explicit_reference_km: &[Option<f32>],
    terminal: &[bool],
    sources_first: &[usize],
) -> Vec<Option<f32>> {
    assert_eq!(drainage_dir.len(), explicit_reference_km.len());
    assert_eq!(drainage_dir.len(), terminal.len());
    assert_eq!(drainage_dir.len(), sources_first.len());
    let mut result = vec![None; drainage_dir.len()];
    for &cell in sources_first.iter().rev() {
        result[cell] = if terminal[cell] {
            None
        } else if let Some(elevation_km) = explicit_reference_km[cell] {
            Some(elevation_km)
        } else {
            drainage_dir[cell].and_then(|downstream| result[downstream])
        };
    }
    result
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn smoothstep(lo: f32, hi: f32, value: f32) -> f32 {
    let t = ((value - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cover_partition_is_exclusive_and_closes() {
        for cover in [0.0, 0.2, 0.7, 1.0] {
            for woody in [0.0, 0.4, 1.0] {
                for wet in [0.0, 0.3, 1.0] {
                    let fractions = partition_cover(cover, woody, wet);
                    assert!((fractions.terrestrial_sum() - 1.0).abs() <= 2e-6);
                    for value in [
                        fractions.bare,
                        fractions.herbaceous,
                        fractions.woody,
                        fractions.wetland,
                    ] {
                        assert!(value.is_finite() && (0.0..=1.0).contains(&value));
                    }
                }
            }
        }
    }

    #[test]
    fn submerged_cells_have_no_terrestrial_cover() {
        let cell = cover_response(0.7, 1.0, 1.0, 1.0, None, true);
        assert_eq!(cell.fractions.terrestrial_sum(), 0.0);
        assert_eq!(cell.thermal_opportunity, 0.0);
        assert_eq!(cell.relative_water_limitation, 0.0);
        assert_eq!(cell.drainage_saturation, 0.0);
        assert_eq!(cell.growth_opportunity, 0.0);
        assert_eq!(cell.vegetation_cover, 0.0);
        assert_eq!(cell.woody_share, 0.0);
        assert_eq!(cell.height_above_drainage_km, None);
    }

    #[test]
    fn planetary_wetness_survives_without_world_mean_cancellation() {
        let dry = cover_response(0.65, 0.45, 0.5, 0.0, None, false);
        let normal = cover_response(0.65, 0.45, 1.0, 0.0, None, false);
        let wet = cover_response(0.65, 0.45, 2.0, 0.0, None, false);
        assert!(dry.relative_water_limitation > normal.relative_water_limitation);
        assert!(normal.relative_water_limitation > wet.relative_water_limitation);
        assert!(dry.vegetation_cover < normal.vegetation_cover);
        assert!(normal.vegetation_cover < wet.vegetation_cover);
    }

    #[test]
    fn absolute_dry_world_does_not_manufacture_riparian_water() {
        let dry_channel = cover_response(0.65, 1.0, 0.0, 0.35, Some(0.0), false);
        assert_eq!(dry_channel.relative_water_limitation, 1.0);
        assert_eq!(dry_channel.vegetation_cover, 0.0);
        assert_eq!(dry_channel.fractions.wetland, 0.0);
    }

    #[test]
    fn precipitation_pattern_and_drainage_wetness_have_causal_signs() {
        let dry = cover_response(0.65, 0.15, 1.0, 0.0, None, false);
        let rainy = cover_response(0.65, 0.75, 1.0, 0.0, None, false);
        let riparian = cover_response(0.65, 0.15, 1.0, 0.9, Some(0.01), false);
        assert!(rainy.relative_water_limitation < dry.relative_water_limitation);
        assert!(rainy.vegetation_cover > dry.vegetation_cover);
        assert!(riparian.vegetation_cover > dry.vegetation_cover);
        assert!(riparian.fractions.wetland > dry.fractions.wetland);
    }

    #[test]
    fn altitude_has_no_second_direct_cutoff_after_temperature() {
        // Elevation is deliberately absent from the response kernel. A lapse
        // correction changes the result through temperature only.
        let warm = cover_response(0.55, 0.8, 1.0, 0.0, None, false);
        let lapse_cold = cover_response(0.05, 0.8, 1.0, 0.0, None, false);
        assert!(warm.thermal_opportunity > lapse_cold.thermal_opportunity);
        assert!(warm.vegetation_cover > lapse_cold.vegetation_cover);
    }

    #[test]
    fn geometric_area_routes_conservatively() {
        let drainage = [Some(1), Some(2), Some(3), None];
        let local = [1.0, 2.0, 3.0, 4.0];
        let (area, order) = geometric_contributing_area(&drainage, &local);
        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(area, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn adaptive_large_cell_cannot_self_anchor_without_upstream_catchment() {
        assert!(!qualifies_as_drainage_reference(10_000.0, 10_000.0));
        assert!(qualifies_as_drainage_reference(40_000.0, 10_000.0));
        assert!(!qualifies_as_drainage_reference(1_999.0, 100.0));
        assert!(qualifies_as_drainage_reference(2_000.0, 100.0));
    }

    #[test]
    fn downstream_reference_distinguishes_bluff_from_floodplain() {
        // Cells 0 and 1 drain to the same channel reference. HAND is therefore
        // controlled by their height above that reference, not Euclidean river
        // distance: cell 0 is the bluff, cell 1 the floodplain.
        let drainage = [Some(2), Some(2), Some(3), None];
        let explicit_reference = [None, None, Some(0.0), Some(-0.1)];
        let terminal = [false; 4];
        let (_, order) = geometric_contributing_area(&drainage, &[1.0; 4]);
        let refs =
            downstream_reference_elevations(&drainage, &explicit_reference, &terminal, &order);
        assert_eq!(refs, vec![Some(0.0), Some(0.0), Some(0.0), Some(-0.1)]);
        let elevation_km = [0.40f32, 0.01, 0.0, -0.1];
        let bluff_hand = (elevation_km[0] - refs[0].unwrap()).max(0.0);
        let plain_hand = (elevation_km[1] - refs[1].unwrap()).max(0.0);
        assert!(bluff_hand > HAND_SATURATION_SCALE_KM);
        assert!(plain_hand < HAND_SATURATION_SCALE_KM);
    }

    #[test]
    fn channel_self_reference_and_terminal_dry_basin_are_explicit() {
        let drainage = [Some(1), Some(2), Some(3), None];
        let explicit_reference = [Some(0.3), None, None, Some(0.0)];
        let terminal = [false, false, true, false];
        let (_, order) = geometric_contributing_area(&drainage, &[1.0; 4]);
        let refs =
            downstream_reference_elevations(&drainage, &explicit_reference, &terminal, &order);
        assert_eq!(refs[0], Some(0.3));
        assert_eq!(refs[1], None);
        assert_eq!(refs[2], None);
        assert_eq!(refs[3], Some(0.0));
    }

    #[test]
    fn graph_permutation_preserves_contributing_area_and_reference_identity() {
        let drainage = [Some(2), Some(2), Some(3), None];
        let local = [2.0, 5.0, 3.0, 7.0];
        let explicit_reference = [None, None, Some(0.2), Some(0.0)];
        let terminal = [false; 4];
        let (area, order) = geometric_contributing_area(&drainage, &local);
        let refs =
            downstream_reference_elevations(&drainage, &explicit_reference, &terminal, &order);

        // new index -> old index
        let permutation = [2usize, 0, 3, 1];
        let mut old_to_new = [0usize; 4];
        for (new, &old) in permutation.iter().enumerate() {
            old_to_new[old] = new;
        }
        let permuted_drainage: Vec<Option<usize>> = permutation
            .iter()
            .map(|&old| drainage[old].map(|downstream| old_to_new[downstream]))
            .collect();
        let permuted_local: Vec<f64> = permutation.iter().map(|&old| local[old]).collect();
        let permuted_reference: Vec<Option<f32>> = permutation
            .iter()
            .map(|&old| explicit_reference[old])
            .collect();
        let permuted_terminal: Vec<bool> = permutation.iter().map(|&old| terminal[old]).collect();
        let (permuted_area, permuted_order) =
            geometric_contributing_area(&permuted_drainage, &permuted_local);
        let permuted_refs = downstream_reference_elevations(
            &permuted_drainage,
            &permuted_reference,
            &permuted_terminal,
            &permuted_order,
        );
        for (new, &old) in permutation.iter().enumerate() {
            assert_eq!(permuted_area[new], area[old]);
            assert_eq!(permuted_refs[new], refs[old]);
        }
    }

    #[test]
    fn hand_saturation_is_continuous_bounded_and_decreasing() {
        let flat_plain = saturation_from_hand(0.0, false);
        let low_plain = saturation_from_hand(0.01, false);
        let slightly_higher = saturation_from_hand(0.011, false);
        let bluff = saturation_from_hand(0.4, false);
        assert_eq!(flat_plain, MAX_DRAINAGE_SATURATED_FRACTION);
        assert!(flat_plain > low_plain);
        assert!(low_plain > slightly_higher);
        assert!(slightly_higher > bluff);
        assert!((low_plain - slightly_higher) < 0.02);
        assert!((0.0..=MAX_DRAINAGE_SATURATED_FRACTION).contains(&bluff));
    }

    #[test]
    fn channel_reference_does_not_claim_unresolved_floodplain_occupancy() {
        // A reference cell may be a steep, narrow channel. Catchment membership
        // alone supplies no defensible subcell channel/floodplain width. A
        // neighboring low-HAND plain can be wet; the reference cell remains
        // unclassified until a width/valley-floor owner exists.
        assert_eq!(saturation_from_hand(0.0, true), 0.0);
        assert_eq!(
            saturation_from_hand(0.0, false),
            MAX_DRAINAGE_SATURATED_FRACTION
        );
    }

    #[test]
    #[should_panic(expected = "acyclic")]
    fn receiver_cycle_is_an_invariant_failure() {
        let _ = geometric_contributing_area(&[Some(1), Some(0)], &[1.0, 1.0]);
    }
}
