//! On-demand physical/access substrate for Consequential Geography V0.
//!
//! This module owns raw, inspectable access components and one bounded aggregate
//! site-selection prior. It deliberately stops before route-network generation
//! and keeps authored combination factors visible.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::Serialize;

use super::{
    elevation_to_km, solid_angle_to_km2, Hydrology, LivingSurfaceSemantics, RiverSelection,
    RiverThresholdPolicy, SemanticWaterKind, Tessellation, WaterBodySemantics, PLANET_RADIUS_KM,
};

#[derive(Clone, Copy, Debug, Serialize)]
pub struct TraversalConfig {
    /// Generalized-km cost per kilometre climbed.
    uphill_penalty: f32,
    /// Generalized-km cost per kilometre descended.
    downhill_penalty: f32,
}

impl TraversalConfig {
    pub fn new(uphill_penalty: f32, downhill_penalty: f32) -> Result<Self, &'static str> {
        if !uphill_penalty.is_finite()
            || !downhill_penalty.is_finite()
            || downhill_penalty < 0.0
            || uphill_penalty < downhill_penalty
        {
            return Err("traversal penalties must be finite with uphill >= downhill >= 0");
        }
        Ok(Self {
            uphill_penalty,
            downhill_penalty,
        })
    }

    pub fn uphill_penalty(self) -> f32 {
        self.uphill_penalty
    }

    pub fn downhill_penalty(self) -> f32 {
        self.downhill_penalty
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DirectedEdgeCost {
    pub from: usize,
    pub to: usize,
    pub distance_km: f32,
    pub elevation_change_km: f32,
    pub ascent_km: f32,
    pub descent_km: f32,
    pub signed_grade: f32,
    pub generalized_cost_km: f32,
    pub touches_drainage_repair: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FreshwaterSourceKind {
    SelectedRiver,
    ProperLakeShore,
    SelectedRiverAndLakeShore,
}

#[derive(Clone, Debug, Serialize)]
pub struct ConsequentialGeographyComponents {
    /// Land-only, direction-neutral generalized access burden to a selected
    /// river or proper-lake shore. `None` means no source is reachable.
    pub freshwater_access_generalized_km: Vec<Option<f32>>,
    /// Land-only, direction-neutral generalized access burden to ocean coast.
    pub coast_access_generalized_km: Vec<Option<f32>>,
    /// Exact source masks used by the two access fields.
    pub freshwater_source: Vec<bool>,
    pub freshwater_source_kind: Vec<Option<FreshwaterSourceKind>>,
    pub coast_source: Vec<bool>,
    /// River-selection provenance for the freshwater source mask.
    pub aggregate_river_policy: RiverThresholdPolicy,
    /// Exact accepted Living Surface vegetation-cover opportunity, not yield or
    /// carrying capacity.
    pub relative_living_opportunity: Vec<f32>,
    pub drainage_saturation: Vec<f32>,
    pub relative_water_limitation: Vec<f32>,
    /// Provenance only: these cells use drainage-integrated effective terrain.
    pub drainage_repaired: Vec<bool>,
    pub traversal: TraversalConfig,
}

/// Disclosed authored prior for one bounded aggregate site configuration.
///
/// The caller must choose these values; there is intentionally no product
/// default before the site model has been evaluated on representative worlds.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct SiteSelectionConfig {
    /// Authored board/cartography budget, constrained to the V0 contract's
    /// 12--30 sites rather than interpreted as a population estimate.
    pub site_count: usize,
    /// Maximum number of expensive catchments evaluated after cheap viability
    /// and spatial preselection.
    pub candidate_pool_size: usize,
    /// Hard cap on total cells visited across evaluated catchment searches.
    pub maximum_total_catchment_cell_visits: usize,
    /// Great-circle separation required between admitted sites.
    pub minimum_site_spacing_km: f32,
    /// Smaller great-circle separation used to diversify the candidate pool.
    pub candidate_spacing_km: f32,
    /// Maximum direction-neutral generalized cost included in a catchment.
    pub catchment_budget_generalized_km: f32,
    /// Hard maximum access burden to selected rivers or proper lakes.
    pub freshwater_access_limit_generalized_km: f32,
    /// Hard minimum accepted local vegetation-cover opportunity.
    pub minimum_local_living_opportunity: f32,
    /// Hard maximum robust mean absolute grade over adjacent land edges, after
    /// removing the single steepest edge when at least three are available.
    pub maximum_local_trimmed_mean_grade: f32,
    /// Hard minimum linearly weighted terrestrial catchment area.
    pub minimum_effective_catchment_area_km2: f32,
    /// Generalized-cost scale over which ocean access supplies a bounded bonus.
    pub coast_access_scale_generalized_km: f32,
    /// Maximum proportional coast bonus. Zero ablates coast preference.
    pub coast_bonus: f32,
}

impl SiteSelectionConfig {
    pub fn validate(self) -> Result<Self, &'static str> {
        if !(12..=30).contains(&self.site_count) {
            return Err("site count must be within the disclosed V0 budget of 12..=30");
        }
        if self.candidate_pool_size < self.site_count {
            return Err("candidate pool must be at least as large as the requested site count");
        }
        if self.candidate_pool_size > 512
            || self.maximum_total_catchment_cell_visits < self.candidate_pool_size
            || self.maximum_total_catchment_cell_visits > 10_000_000
        {
            return Err("site evaluation must stay within the V0 candidate and retained-cell caps");
        }
        if !self.minimum_site_spacing_km.is_finite()
            || self.minimum_site_spacing_km <= 0.0
            || !self.candidate_spacing_km.is_finite()
            || self.candidate_spacing_km <= 0.0
            || self.candidate_spacing_km > self.minimum_site_spacing_km
        {
            return Err("site spacing must be finite with 0 < candidate <= admitted spacing");
        }
        if !self.catchment_budget_generalized_km.is_finite()
            || self.catchment_budget_generalized_km <= 0.0
            || self.catchment_budget_generalized_km > 2_000.0
            || !self.freshwater_access_limit_generalized_km.is_finite()
            || self.freshwater_access_limit_generalized_km <= 0.0
            || self.freshwater_access_limit_generalized_km > 2_000.0
            || !self.minimum_local_living_opportunity.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_local_living_opportunity)
            || !self.maximum_local_trimmed_mean_grade.is_finite()
            || self.maximum_local_trimmed_mean_grade <= 0.0
            || !self.minimum_effective_catchment_area_km2.is_finite()
            || self.minimum_effective_catchment_area_km2 <= 0.0
            || !self.coast_access_scale_generalized_km.is_finite()
            || self.coast_access_scale_generalized_km <= 0.0
            || !self.coast_bonus.is_finite()
            || !(0.0..=1.0).contains(&self.coast_bonus)
        {
            return Err("site opportunity scales must be positive and coast bonus within 0..=1");
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AggregateSite {
    /// Deterministic identity within this generated selection, ordered by
    /// greedy admission rather than claimed as persistent world identity.
    pub id: usize,
    pub anchor_cell: usize,
    pub candidate_preselection_rank: usize,
    pub freshwater_source: bool,
    pub nearest_freshwater_source_cell: usize,
    pub nearest_freshwater_source_kind: FreshwaterSourceKind,
    pub coast_source: bool,
    pub freshwater_access_generalized_km: f32,
    pub coast_access_generalized_km: Option<f32>,
    pub local_living_opportunity: f32,
    pub local_trimmed_mean_absolute_grade: f32,
    pub reachable_catchment_area_km2: f32,
    pub effective_catchment_area_km2: f32,
    pub catchment_visited_cell_count: usize,
    pub maximum_catchment_cost_generalized_km: f32,
    /// Area-equivalent accepted vegetation opportunity under the disclosed
    /// linear catchment-access kernel; not yield or carrying capacity.
    pub accessible_living_opportunity_km2: f32,
    pub accessible_mean_living_opportunity: f32,
    /// Portion not already claimed by earlier admitted site catchments.
    pub unclaimed_living_opportunity_km2: f32,
    pub claimed_opportunity_fraction: f32,
    pub drainage_repair_effective_area_fraction: f32,
    pub anchor_drainage_repaired: bool,
    pub freshwater_factor: f32,
    pub terrain_factor: f32,
    pub coast_proximity_factor: f32,
    pub coast_multiplier: f32,
    pub selection_score_km2: f32,
    /// Nearest earlier admitted site at admission; `None` for the first site.
    pub admission_nearest_site_distance_km: Option<f32>,
    pub required_spacing_km: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct AggregateSiteSelection {
    pub config: SiteSelectionConfig,
    pub traversal: TraversalConfig,
    pub aggregate_river_policy: RiverThresholdPolicy,
    pub eligible_cell_count: usize,
    pub local_maximum_candidate_count: usize,
    pub preselected_candidate_count: usize,
    pub catchment_rejected_count: usize,
    pub candidate_pool_count: usize,
    pub catchment_search_count: usize,
    pub catchment_visited_cell_count: usize,
    pub sites: Vec<AggregateSite>,
    pub site_shortfall: usize,
    pub spacing_suppressed_candidate_count: usize,
    pub stop_reason: SiteSelectionStopReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SiteSelectionStopReason {
    RequestedCountReached,
    CandidateOrSpacingExhausted,
    NoUnclaimedOpportunity,
}

impl ConsequentialGeographyComponents {
    pub fn build(
        tessellation: &Tessellation,
        hydrology: &Hydrology,
        water: &WaterBodySemantics,
        rivers: &RiverSelection,
        living: &LivingSurfaceSemantics,
        traversal: TraversalConfig,
    ) -> Result<Self, &'static str> {
        let n = tessellation.num_cells();
        if hydrology.elevation.len() != n
            || hydrology.is_ocean.len() != n
            || hydrology.basin_id.len() != n
            || hydrology.cell_water_body.len() != n
            || water.cell_body.len() != n
            || rivers.all_cells.len() != n
            || living.cells.len() != n
        {
            return Err("consequential-geography input lengths must match tessellation");
        }
        if hydrology.elevation.iter().any(|value| !value.is_finite()) {
            return Err("consequential-geography terrain must be finite");
        }
        if hydrology
            .cell_water_body
            .iter()
            .flatten()
            .any(|&body| body >= hydrology.water_bodies.len())
        {
            return Err("consequential-geography hydrology water-body index is out of range");
        }
        if water
            .cell_body
            .iter()
            .flatten()
            .any(|&body| body >= water.bodies.len())
        {
            return Err("consequential-geography water-body index is out of range");
        }
        if living.cells.iter().any(|cell| {
            [
                cell.vegetation_cover,
                cell.drainage_saturation,
                cell.relative_water_limitation,
            ]
            .into_iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        }) {
            return Err("consequential-geography living components must be finite fractions");
        }

        let submerged: Vec<bool> = (0..n).map(|cell| hydrology.is_submerged(cell)).collect();
        let (freshwater_sources, freshwater_source_kind, coast_sources) =
            source_masks(tessellation, &submerged, water, &rivers.all_cells);

        let freshwater_access_generalized_km = access_costs(
            tessellation,
            &hydrology.elevation,
            &submerged,
            &freshwater_sources,
            traversal,
        )?;
        let coast_access_generalized_km = access_costs(
            tessellation,
            &hydrology.elevation,
            &submerged,
            &coast_sources,
            traversal,
        )?;

        Ok(Self {
            freshwater_access_generalized_km,
            coast_access_generalized_km,
            freshwater_source: freshwater_sources,
            freshwater_source_kind,
            coast_source: coast_sources,
            aggregate_river_policy: rivers.policy,
            relative_living_opportunity: living
                .cells
                .iter()
                .map(|cell| cell.vegetation_cover)
                .collect(),
            drainage_saturation: living
                .cells
                .iter()
                .map(|cell| cell.drainage_saturation)
                .collect(),
            relative_water_limitation: living
                .cells
                .iter()
                .map(|cell| cell.relative_water_limitation)
                .collect(),
            drainage_repaired: (0..n)
                .map(|cell| hydrology.was_lowered_by_integration(cell))
                .collect(),
            traversal,
        })
    }

    /// Select one deterministic, spatially competing aggregate site
    /// configuration from the raw components.
    ///
    /// Freshwater access and local grade are hard gates. Remaining factors are
    /// explicit and bounded: freshwater and terrain each retain a factor in
    /// `[0.5, 1]`, coast supplies at most `config.coast_bonus`, and accepted
    /// living opportunity is integrated with a linear travel-cost kernel.
    pub fn select_sites(
        &self,
        tessellation: &Tessellation,
        hydrology: &Hydrology,
        config: SiteSelectionConfig,
    ) -> Result<AggregateSiteSelection, &'static str> {
        let config = config.validate()?;
        let n = tessellation.num_cells();
        if hydrology.elevation.len() != n
            || hydrology.is_ocean.len() != n
            || hydrology.basin_id.len() != n
            || hydrology.cell_water_body.len() != n
            || self.freshwater_access_generalized_km.len() != n
            || self.coast_access_generalized_km.len() != n
            || self.freshwater_source.len() != n
            || self.freshwater_source_kind.len() != n
            || self.coast_source.len() != n
            || self.relative_living_opportunity.len() != n
            || self.drainage_saturation.len() != n
            || self.relative_water_limitation.len() != n
            || self.drainage_repaired.len() != n
        {
            return Err("site-selection inputs must match tessellation");
        }
        if hydrology.elevation.iter().any(|value| !value.is_finite()) {
            return Err("site-selection terrain must be finite");
        }
        if hydrology
            .cell_water_body
            .iter()
            .flatten()
            .any(|&body| body >= hydrology.water_bodies.len())
        {
            return Err("site-selection hydrology water-body index is out of range");
        }
        if self
            .freshwater_access_generalized_km
            .iter()
            .chain(&self.coast_access_generalized_km)
            .flatten()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("site-selection access burdens must be finite and non-negative");
        }
        if self
            .relative_living_opportunity
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("site-selection living opportunity must be a finite fraction");
        }
        if self
            .freshwater_source
            .iter()
            .zip(&self.freshwater_source_kind)
            .any(|(&source, kind)| source != kind.is_some())
        {
            return Err("site-selection freshwater source mask and kind disagree");
        }
        let submerged: Vec<bool> = (0..n).map(|cell| hydrology.is_submerged(cell)).collect();
        select_sites_from_fields(tessellation, &hydrology.elevation, &submerged, self, config)
    }
}

#[derive(Clone, Copy, Debug)]
struct RankedCandidate {
    cell: usize,
    preliminary_score: f32,
}

#[derive(Clone, Copy, Debug)]
struct CatchmentCell {
    cell: usize,
    access_weight: f32,
    living_contribution_km2: f32,
}

#[derive(Clone, Debug)]
struct EvaluatedCandidate {
    anchor_cell: usize,
    preselection_rank: usize,
    local_trimmed_mean_absolute_grade: f32,
    freshwater_access_generalized_km: f32,
    freshwater_factor: f32,
    terrain_factor: f32,
    coast_proximity_factor: f32,
    coast_multiplier: f32,
    reachable_area_km2: f32,
    effective_area_km2: f32,
    visited_cell_count: usize,
    maximum_cost_generalized_km: f32,
    accessible_living_opportunity_km2: f32,
    accessible_mean_living_opportunity: f32,
    drainage_repair_effective_area_fraction: f32,
    catchment: Vec<CatchmentCell>,
}

fn select_sites_from_fields(
    tessellation: &Tessellation,
    elevation: &[f32],
    submerged: &[bool],
    components: &ConsequentialGeographyComponents,
    config: SiteSelectionConfig,
) -> Result<AggregateSiteSelection, &'static str> {
    let n = tessellation.num_cells();
    let mut proposal_scores = vec![f32::NEG_INFINITY; n];
    let mut eligible_cell_count = 0;
    for cell in 0..n {
        if submerged[cell] {
            continue;
        }
        let Some(freshwater_access) = components.freshwater_access_generalized_km[cell] else {
            continue;
        };
        if freshwater_access > config.freshwater_access_limit_generalized_km {
            continue;
        }
        let local_living = components.relative_living_opportunity[cell];
        if local_living < config.minimum_local_living_opportunity {
            continue;
        }
        let local_grade =
            local_trimmed_mean_absolute_land_grade(tessellation, elevation, submerged, cell)?;
        if local_grade > config.maximum_local_trimmed_mean_grade {
            continue;
        }
        let coast_proximity = coast_proximity_factor(
            components.coast_access_generalized_km[cell],
            config.coast_access_scale_generalized_km,
        );
        let coast_multiplier = 1.0 + config.coast_bonus * coast_proximity;
        let living_margin = if config.minimum_local_living_opportunity < 1.0 {
            (local_living - config.minimum_local_living_opportunity)
                / (1.0 - config.minimum_local_living_opportunity)
        } else {
            1.0
        };
        let freshwater_margin =
            1.0 - freshwater_access / config.freshwater_access_limit_generalized_km;
        let terrain_margin = 1.0 - local_grade / config.maximum_local_trimmed_mean_grade;
        proposal_scores[cell] = living_margin
            .min(freshwater_margin)
            .min(terrain_margin)
            .max(0.0)
            * coast_multiplier;
        eligible_cell_count += 1;
    }

    let mut ranked = Vec::new();
    for cell in 0..n {
        let score = proposal_scores[cell];
        if !score.is_finite() {
            continue;
        }
        let is_local_maximum = tessellation.neighbors(cell).iter().all(|&neighbor| {
            proposal_scores[neighbor] < score
                || (proposal_scores[neighbor].to_bits() == score.to_bits() && neighbor > cell)
        });
        if is_local_maximum {
            ranked.push(RankedCandidate {
                cell,
                preliminary_score: score,
            });
        }
    }
    let local_maximum_candidate_count = ranked.len();
    ranked.sort_unstable_by(|a, b| {
        b.preliminary_score
            .total_cmp(&a.preliminary_score)
            .then_with(|| a.cell.cmp(&b.cell))
    });

    let mut pool_cells = Vec::with_capacity(config.candidate_pool_size);
    for (rank, candidate) in ranked.into_iter().enumerate() {
        if pool_cells.iter().all(|&(accepted, _)| {
            physical_distance_km(tessellation, candidate.cell, accepted)
                >= config.candidate_spacing_km
        }) {
            pool_cells.push((candidate.cell, rank + 1));
            if pool_cells.len() == config.candidate_pool_size {
                break;
            }
        }
    }

    let preselected_candidate_count = pool_cells.len();
    let areas = tessellation.cell_areas_ref();
    let mut distance_scratch = vec![f32::INFINITY; n];
    let mut candidates = Vec::with_capacity(pool_cells.len());
    let mut catchment_visited_cell_count = 0usize;
    for (anchor_cell, preselection_rank) in pool_cells {
        let local_grade = local_trimmed_mean_absolute_land_grade(
            tessellation,
            elevation,
            submerged,
            anchor_cell,
        )?;
        let freshwater_access = components.freshwater_access_generalized_km[anchor_cell]
            .ok_or("preselected site lost freshwater access")?;
        let freshwater_factor = limiting_factor(
            freshwater_access,
            config.freshwater_access_limit_generalized_km,
        );
        let terrain_factor = limiting_factor(local_grade, config.maximum_local_trimmed_mean_grade);
        let coast_proximity = coast_proximity_factor(
            components.coast_access_generalized_km[anchor_cell],
            config.coast_access_scale_generalized_km,
        );
        let coast_multiplier = 1.0 + config.coast_bonus * coast_proximity;
        let catchment = bounded_catchment(
            tessellation,
            elevation,
            submerged,
            &components.relative_living_opportunity,
            &components.drainage_repaired,
            areas,
            anchor_cell,
            config.catchment_budget_generalized_km,
            components.traversal,
            &mut distance_scratch,
        )?;
        catchment_visited_cell_count += catchment.cells.len();
        if catchment_visited_cell_count > config.maximum_total_catchment_cell_visits {
            return Err("site catchments exceeded the configured cell-visit budget");
        }
        if catchment.effective_area_km2 < config.minimum_effective_catchment_area_km2 {
            continue;
        }
        candidates.push(EvaluatedCandidate {
            anchor_cell,
            preselection_rank,
            local_trimmed_mean_absolute_grade: local_grade,
            freshwater_access_generalized_km: freshwater_access,
            freshwater_factor,
            terrain_factor,
            coast_proximity_factor: coast_proximity,
            coast_multiplier,
            reachable_area_km2: catchment.reachable_area_km2,
            effective_area_km2: catchment.effective_area_km2,
            visited_cell_count: catchment.cells.len(),
            maximum_cost_generalized_km: catchment.maximum_cost_generalized_km,
            accessible_living_opportunity_km2: catchment.living_opportunity_km2,
            accessible_mean_living_opportunity: catchment.accessible_mean_living_opportunity,
            drainage_repair_effective_area_fraction: catchment
                .drainage_repair_effective_area_fraction,
            catchment: catchment.cells,
        });
    }

    let candidate_pool_count = candidates.len();
    let catchment_rejected_count = preselected_candidate_count - candidate_pool_count;
    let mut admitted = vec![false; candidate_pool_count];
    let mut claimed_access_weight = vec![0.0f32; n];
    let mut sites = Vec::with_capacity(config.site_count);
    let mut stopped_for_no_unclaimed_opportunity = false;
    while sites.len() < config.site_count {
        let mut best: Option<(usize, f32, f32, f32, Option<f32>)> = None;
        for (index, candidate) in candidates.iter().enumerate() {
            if admitted[index] {
                continue;
            }
            let nearest = sites
                .iter()
                .map(|site: &AggregateSite| {
                    physical_distance_km(tessellation, candidate.anchor_cell, site.anchor_cell)
                })
                .min_by(f32::total_cmp);
            if nearest.is_some_and(|distance| distance < config.minimum_site_spacing_km) {
                continue;
            }
            let unclaimed =
                unclaimed_living_opportunity(&candidate.catchment, &claimed_access_weight);
            let claimed_fraction = if candidate.accessible_living_opportunity_km2 > 0.0 {
                (1.0 - unclaimed / candidate.accessible_living_opportunity_km2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let score = unclaimed
                * candidate.freshwater_factor
                * candidate.terrain_factor
                * candidate.coast_multiplier;
            let replace = best
                .as_ref()
                .is_none_or(|&(best_index, best_score, _, _, _)| {
                    score > best_score
                        || (score.to_bits() == best_score.to_bits()
                            && candidate.anchor_cell < candidates[best_index].anchor_cell)
                });
            if replace {
                best = Some((index, score, unclaimed, claimed_fraction, nearest));
            }
        }

        let Some((index, score, unclaimed, claimed_fraction, nearest)) = best else {
            break;
        };
        if !score.is_finite() || score <= 0.0 {
            stopped_for_no_unclaimed_opportunity = true;
            break;
        }
        admitted[index] = true;
        let candidate = &candidates[index];
        let freshwater_search_tolerance =
            (candidate.freshwater_access_generalized_km.abs() * 1e-5).max(1e-3);
        let (nearest_freshwater_source_cell, nearest_freshwater_source_kind) =
            nearest_freshwater_source(
                tessellation,
                elevation,
                submerged,
                &components.freshwater_source_kind,
                candidate.anchor_cell,
                candidate.freshwater_access_generalized_km + freshwater_search_tolerance,
                components.traversal,
                &mut distance_scratch,
            )?
            .ok_or("admitted site has no freshwater source within its accepted burden")?;
        for entry in &candidate.catchment {
            claimed_access_weight[entry.cell] =
                claimed_access_weight[entry.cell].max(entry.access_weight);
        }
        sites.push(AggregateSite {
            id: sites.len(),
            anchor_cell: candidate.anchor_cell,
            candidate_preselection_rank: candidate.preselection_rank,
            freshwater_source: components.freshwater_source[candidate.anchor_cell],
            nearest_freshwater_source_cell,
            nearest_freshwater_source_kind,
            coast_source: components.coast_source[candidate.anchor_cell],
            freshwater_access_generalized_km: candidate.freshwater_access_generalized_km,
            coast_access_generalized_km: components.coast_access_generalized_km
                [candidate.anchor_cell],
            local_living_opportunity: components.relative_living_opportunity[candidate.anchor_cell],
            local_trimmed_mean_absolute_grade: candidate.local_trimmed_mean_absolute_grade,
            reachable_catchment_area_km2: candidate.reachable_area_km2,
            effective_catchment_area_km2: candidate.effective_area_km2,
            catchment_visited_cell_count: candidate.visited_cell_count,
            maximum_catchment_cost_generalized_km: candidate.maximum_cost_generalized_km,
            accessible_living_opportunity_km2: candidate.accessible_living_opportunity_km2,
            accessible_mean_living_opportunity: candidate.accessible_mean_living_opportunity,
            unclaimed_living_opportunity_km2: unclaimed,
            claimed_opportunity_fraction: claimed_fraction,
            drainage_repair_effective_area_fraction: candidate
                .drainage_repair_effective_area_fraction,
            anchor_drainage_repaired: components.drainage_repaired[candidate.anchor_cell],
            freshwater_factor: candidate.freshwater_factor,
            terrain_factor: candidate.terrain_factor,
            coast_proximity_factor: candidate.coast_proximity_factor,
            coast_multiplier: candidate.coast_multiplier,
            selection_score_km2: score,
            admission_nearest_site_distance_km: nearest,
            required_spacing_km: config.minimum_site_spacing_km,
        });
    }

    let spacing_suppressed_candidate_count = candidates
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            !admitted[*index]
                && sites.iter().any(|site| {
                    physical_distance_km(tessellation, candidate.anchor_cell, site.anchor_cell)
                        < config.minimum_site_spacing_km
                })
        })
        .count();
    let stop_reason = if sites.len() == config.site_count {
        SiteSelectionStopReason::RequestedCountReached
    } else if stopped_for_no_unclaimed_opportunity {
        SiteSelectionStopReason::NoUnclaimedOpportunity
    } else {
        SiteSelectionStopReason::CandidateOrSpacingExhausted
    };
    Ok(AggregateSiteSelection {
        config,
        traversal: components.traversal,
        aggregate_river_policy: components.aggregate_river_policy,
        eligible_cell_count,
        local_maximum_candidate_count,
        preselected_candidate_count,
        catchment_rejected_count,
        candidate_pool_count,
        catchment_search_count: preselected_candidate_count,
        catchment_visited_cell_count,
        site_shortfall: config.site_count - sites.len(),
        sites,
        spacing_suppressed_candidate_count,
        stop_reason,
    })
}

fn unclaimed_living_opportunity(catchment: &[CatchmentCell], claimed_access_weight: &[f32]) -> f32 {
    catchment
        .iter()
        .map(|entry| {
            if entry.access_weight <= 0.0 {
                return 0.0;
            }
            let marginal_weight =
                (entry.access_weight - claimed_access_weight[entry.cell]).max(0.0);
            entry.living_contribution_km2 * marginal_weight / entry.access_weight
        })
        .sum()
}

#[derive(Clone, Debug)]
struct BoundedCatchment {
    cells: Vec<CatchmentCell>,
    reachable_area_km2: f32,
    effective_area_km2: f32,
    living_opportunity_km2: f32,
    accessible_mean_living_opportunity: f32,
    maximum_cost_generalized_km: f32,
    drainage_repair_effective_area_fraction: f32,
}

#[allow(clippy::too_many_arguments)]
fn bounded_catchment(
    tessellation: &Tessellation,
    elevation: &[f32],
    submerged: &[bool],
    living_opportunity: &[f32],
    drainage_repaired: &[bool],
    areas_steradians: &[f32],
    anchor_cell: usize,
    budget_generalized_km: f32,
    traversal: TraversalConfig,
    distance_scratch: &mut [f32],
) -> Result<BoundedCatchment, &'static str> {
    let mut touched = vec![anchor_cell];
    let mut queue = BinaryHeap::new();
    distance_scratch[anchor_cell] = 0.0;
    queue.push(QueueEntry {
        cost: 0.0,
        cell: anchor_cell,
    });
    while let Some(QueueEntry { cost, cell }) = queue.pop() {
        if cost > distance_scratch[cell] {
            continue;
        }
        for &neighbor in tessellation.neighbors(cell) {
            if submerged[neighbor] {
                continue;
            }
            let edge = symmetric_edge_generalized_cost(
                tessellation,
                elevation,
                cell,
                neighbor,
                traversal,
            )?;
            let next = cost + edge;
            if next <= budget_generalized_km && next < distance_scratch[neighbor] {
                if distance_scratch[neighbor].is_infinite() {
                    touched.push(neighbor);
                }
                distance_scratch[neighbor] = next;
                queue.push(QueueEntry {
                    cost: next,
                    cell: neighbor,
                });
            }
        }
    }

    let mut cells = Vec::with_capacity(touched.len());
    let mut reachable_area_km2 = 0.0f64;
    let mut effective_area_km2 = 0.0f64;
    let mut living_opportunity_km2 = 0.0f64;
    let mut repaired_effective_area_km2 = 0.0f64;
    let mut maximum_cost_generalized_km = 0.0f32;
    for &cell in &touched {
        let cost = distance_scratch[cell];
        maximum_cost_generalized_km = maximum_cost_generalized_km.max(cost);
        let access_weight = (1.0 - cost / budget_generalized_km).clamp(0.0, 1.0);
        let area_km2 = solid_angle_to_km2(areas_steradians[cell]) as f64;
        let effective_area = access_weight as f64 * area_km2;
        let living_contribution = effective_area * living_opportunity[cell] as f64;
        reachable_area_km2 += area_km2;
        effective_area_km2 += effective_area;
        living_opportunity_km2 += living_contribution;
        if drainage_repaired[cell] {
            repaired_effective_area_km2 += effective_area;
        }
        cells.push(CatchmentCell {
            cell,
            access_weight,
            living_contribution_km2: living_contribution as f32,
        });
        distance_scratch[cell] = f32::INFINITY;
    }
    let drainage_repair_effective_area_fraction = if effective_area_km2 > 0.0 {
        (repaired_effective_area_km2 / effective_area_km2).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let accessible_mean_living_opportunity = if effective_area_km2 > 0.0 {
        (living_opportunity_km2 / effective_area_km2).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    Ok(BoundedCatchment {
        cells,
        reachable_area_km2: reachable_area_km2 as f32,
        effective_area_km2: effective_area_km2 as f32,
        living_opportunity_km2: living_opportunity_km2 as f32,
        accessible_mean_living_opportunity,
        maximum_cost_generalized_km,
        drainage_repair_effective_area_fraction,
    })
}

fn limiting_factor(value: f32, hard_limit: f32) -> f32 {
    1.0 - 0.5 * (value / hard_limit).clamp(0.0, 1.0)
}

fn coast_proximity_factor(access: Option<f32>, scale: f32) -> f32 {
    access
        .map(|cost| (1.0 - cost / scale).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

fn local_trimmed_mean_absolute_land_grade(
    tessellation: &Tessellation,
    elevation: &[f32],
    submerged: &[bool],
    cell: usize,
) -> Result<f32, &'static str> {
    let mut grade_sum = 0.0;
    let mut maximum_grade = 0.0f32;
    let mut count = 0usize;
    for &neighbor in tessellation.neighbors(cell) {
        if submerged[neighbor] {
            continue;
        }
        let distance = physical_distance_km(tessellation, cell, neighbor);
        let elevation_change = elevation_to_km(elevation[neighbor] - elevation[cell]);
        let grade = elevation_change.abs() / distance;
        if !grade.is_finite() {
            return Err("site candidate grade must be finite");
        }
        grade_sum += grade;
        maximum_grade = maximum_grade.max(grade);
        count += 1;
    }
    if count == 0 {
        return Ok(f32::INFINITY);
    }
    if count >= 3 {
        Ok((grade_sum - maximum_grade) / (count - 1) as f32)
    } else {
        Ok(grade_sum / count as f32)
    }
}

fn physical_distance_km(tessellation: &Tessellation, from: usize, to: usize) -> f32 {
    let chord = (tessellation.cell_center(from) - tessellation.cell_center(to)).length();
    2.0 * PLANET_RADIUS_KM * (0.5 * chord).clamp(0.0, 1.0).asin()
}

fn symmetric_edge_generalized_cost(
    tessellation: &Tessellation,
    elevation: &[f32],
    from: usize,
    to: usize,
    config: TraversalConfig,
) -> Result<f32, &'static str> {
    let forward = edge_cost_from_elevation(tessellation, elevation, from, to, false, config)?;
    let reverse = edge_cost_from_elevation(tessellation, elevation, to, from, false, config)?;
    Ok(0.5 * (forward.generalized_cost_km + reverse.generalized_cost_km))
}

#[allow(clippy::too_many_arguments)]
fn nearest_freshwater_source(
    tessellation: &Tessellation,
    elevation: &[f32],
    submerged: &[bool],
    source_kind: &[Option<FreshwaterSourceKind>],
    anchor_cell: usize,
    maximum_cost: f32,
    traversal: TraversalConfig,
    distance_scratch: &mut [f32],
) -> Result<Option<(usize, FreshwaterSourceKind)>, &'static str> {
    let mut touched = vec![anchor_cell];
    let mut queue = BinaryHeap::new();
    let mut found = None;
    distance_scratch[anchor_cell] = 0.0;
    queue.push(QueueEntry {
        cost: 0.0,
        cell: anchor_cell,
    });
    while let Some(QueueEntry { cost, cell }) = queue.pop() {
        if cost > distance_scratch[cell] {
            continue;
        }
        if let Some(kind) = source_kind[cell] {
            found = Some((cell, kind));
            break;
        }
        for &neighbor in tessellation.neighbors(cell) {
            if submerged[neighbor] {
                continue;
            }
            let next = cost
                + symmetric_edge_generalized_cost(
                    tessellation,
                    elevation,
                    cell,
                    neighbor,
                    traversal,
                )?;
            if next <= maximum_cost && next < distance_scratch[neighbor] {
                if distance_scratch[neighbor].is_infinite() {
                    touched.push(neighbor);
                }
                distance_scratch[neighbor] = next;
                queue.push(QueueEntry {
                    cost: next,
                    cell: neighbor,
                });
            }
        }
    }
    for cell in touched {
        distance_scratch[cell] = f32::INFINITY;
    }
    Ok(found)
}

fn source_masks(
    tessellation: &Tessellation,
    submerged: &[bool],
    water: &WaterBodySemantics,
    selected_rivers: &[bool],
) -> (Vec<bool>, Vec<Option<FreshwaterSourceKind>>, Vec<bool>) {
    let n = tessellation.num_cells();
    let mut freshwater = vec![false; n];
    let mut freshwater_kind = vec![None; n];
    let mut coast = vec![false; n];
    for cell in 0..n {
        if submerged[cell] {
            continue;
        }
        let river = selected_rivers[cell];
        let mut lake_shore = false;
        for &neighbor in tessellation.neighbors(cell) {
            let Some(body_index) = water.cell_body[neighbor] else {
                continue;
            };
            match water.bodies[body_index].kind {
                SemanticWaterKind::Ocean => coast[cell] = true,
                SemanticWaterKind::Lake => lake_shore = true,
                SemanticWaterKind::Pond => {}
            }
        }
        freshwater_kind[cell] = match (river, lake_shore) {
            (true, true) => Some(FreshwaterSourceKind::SelectedRiverAndLakeShore),
            (true, false) => Some(FreshwaterSourceKind::SelectedRiver),
            (false, true) => Some(FreshwaterSourceKind::ProperLakeShore),
            (false, false) => None,
        };
        freshwater[cell] = freshwater_kind[cell].is_some();
    }
    (freshwater, freshwater_kind, coast)
}

pub fn directed_edge_cost(
    tessellation: &Tessellation,
    hydrology: &Hydrology,
    from: usize,
    to: usize,
    config: TraversalConfig,
) -> Result<DirectedEdgeCost, &'static str> {
    if from >= tessellation.num_cells()
        || to >= tessellation.num_cells()
        || hydrology.elevation.len() != tessellation.num_cells()
        || !tessellation.neighbors(from).contains(&to)
    {
        return Err("directed traversal requires an in-range adjacent cell pair");
    }
    edge_cost_from_elevation(
        tessellation,
        &hydrology.elevation,
        from,
        to,
        hydrology.was_lowered_by_integration(from) || hydrology.was_lowered_by_integration(to),
        config,
    )
}

fn edge_cost_from_elevation(
    tessellation: &Tessellation,
    elevation: &[f32],
    from: usize,
    to: usize,
    touches_drainage_repair: bool,
    config: TraversalConfig,
) -> Result<DirectedEdgeCost, &'static str> {
    let chord = (tessellation.cell_center(from) - tessellation.cell_center(to)).length();
    let distance_km = 2.0 * PLANET_RADIUS_KM * (0.5 * chord).clamp(0.0, 1.0).asin();
    if !distance_km.is_finite() || distance_km <= 0.0 {
        return Err("adjacent traversal edge must have positive finite length");
    }
    let elevation_change_km = elevation_to_km(elevation[to] - elevation[from]);
    if !elevation_change_km.is_finite() {
        return Err("traversal elevation change must be finite");
    }
    let ascent_km = elevation_change_km.max(0.0);
    let descent_km = (-elevation_change_km).max(0.0);
    let generalized_cost_km =
        distance_km + config.uphill_penalty * ascent_km + config.downhill_penalty * descent_km;
    if !generalized_cost_km.is_finite() || generalized_cost_km < distance_km {
        return Err("generalized traversal cost must be finite and at least physical distance");
    }
    Ok(DirectedEdgeCost {
        from,
        to,
        distance_km,
        elevation_change_km,
        ascent_km,
        descent_km,
        signed_grade: elevation_change_km / distance_km,
        generalized_cost_km,
        touches_drainage_repair,
    })
}

#[derive(Clone, Copy, Debug)]
struct QueueEntry {
    cost: f32,
    cell: usize,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits() && self.cell == other.cell
    }
}
impl Eq for QueueEntry {}
impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

fn access_costs(
    tessellation: &Tessellation,
    elevation: &[f32],
    submerged: &[bool],
    sources: &[bool],
    config: TraversalConfig,
) -> Result<Vec<Option<f32>>, &'static str> {
    let n = tessellation.num_cells();
    if elevation.len() != n || submerged.len() != n || sources.len() != n {
        return Err("access-field inputs must match tessellation");
    }
    let mut distances = vec![f32::INFINITY; n];
    let mut queue = BinaryHeap::new();
    for cell in 0..n {
        if sources[cell] && !submerged[cell] {
            distances[cell] = 0.0;
            queue.push(QueueEntry { cost: 0.0, cell });
        }
    }
    while let Some(QueueEntry { cost, cell }) = queue.pop() {
        if cost > distances[cell] {
            continue;
        }
        for &neighbor in tessellation.neighbors(cell) {
            if submerged[neighbor] {
                continue;
            }
            let next = cost
                + symmetric_edge_generalized_cost(tessellation, elevation, cell, neighbor, config)?;
            if next < distances[neighbor] {
                distances[neighbor] = next;
                queue.push(QueueEntry {
                    cost: next,
                    cell: neighbor,
                });
            }
        }
    }
    Ok(distances
        .into_iter()
        .map(|distance| distance.is_finite().then_some(distance))
        .collect())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::world::{Elevation, NoiseLayerData, SemanticWaterBody, WaterBodyId, WaterOutlet};

    fn tessellation(cells: usize) -> Tessellation {
        let mut rng = ChaCha8Rng::seed_from_u64(61_207);
        Tessellation::generate(cells, 0, &mut rng)
    }

    fn config() -> TraversalConfig {
        TraversalConfig::new(12.0, 3.0).unwrap()
    }

    fn site_config() -> SiteSelectionConfig {
        SiteSelectionConfig {
            site_count: 12,
            candidate_pool_size: 64,
            maximum_total_catchment_cell_visits: 1_000_000,
            minimum_site_spacing_km: 1.0,
            candidate_spacing_km: 1.0,
            catchment_budget_generalized_km: 1.0,
            freshwater_access_limit_generalized_km: 100.0,
            minimum_local_living_opportunity: 0.05,
            maximum_local_trimmed_mean_grade: 0.5,
            minimum_effective_catchment_area_km2: 1.0,
            coast_access_scale_generalized_km: 500.0,
            coast_bonus: 0.25,
        }
    }

    fn synthetic_components(living: Vec<f32>) -> ConsequentialGeographyComponents {
        let n = living.len();
        ConsequentialGeographyComponents {
            freshwater_access_generalized_km: vec![Some(0.0); n],
            coast_access_generalized_km: vec![None; n],
            freshwater_source: vec![true; n],
            freshwater_source_kind: vec![Some(FreshwaterSourceKind::SelectedRiver); n],
            coast_source: vec![false; n],
            aggregate_river_policy: RiverThresholdPolicy::default(),
            relative_living_opportunity: living,
            drainage_saturation: vec![0.0; n],
            relative_water_limitation: vec![0.0; n],
            drainage_repaired: vec![false; n],
            traversal: config(),
        }
    }

    fn one_cell_water(
        cell_count: usize,
        cell: usize,
        kind: SemanticWaterKind,
    ) -> WaterBodySemantics {
        let mut cell_body = vec![None; cell_count];
        cell_body[cell] = Some(0);
        WaterBodySemantics {
            bodies: vec![SemanticWaterBody {
                id: WaterBodyId {
                    basin_id: None,
                    anchor_cell: cell,
                },
                kind,
                cells: vec![cell],
                area_km2: 1.0,
                surface_elevation_km: 0.0,
                max_depth_km: 1.0,
                outlet: WaterOutlet::Terminal,
            }],
            cell_body,
        }
    }

    #[test]
    fn traversal_config_rejects_invalid_penalty_order_and_values() {
        assert!(TraversalConfig::new(2.0, 3.0).is_err());
        assert!(TraversalConfig::new(2.0, -1.0).is_err());
        assert!(TraversalConfig::new(f32::NAN, 0.0).is_err());
        let valid = TraversalConfig::new(3.0, 2.0).unwrap();
        assert_eq!(valid.uphill_penalty(), 3.0);
        assert_eq!(valid.downhill_penalty(), 2.0);
    }

    #[test]
    fn site_config_rejects_an_undisclosed_site_budget() {
        let mut invalid = site_config();
        invalid.site_count = 31;
        assert!(invalid.validate().is_err());
        invalid.site_count = 12;
        invalid.candidate_pool_size = 11;
        assert!(invalid.validate().is_err());
        invalid.candidate_pool_size = 513;
        invalid.maximum_total_catchment_cell_visits = 1_000_000;
        assert!(invalid.validate().is_err());
        invalid.candidate_pool_size = 64;
        invalid.catchment_budget_generalized_km = 0.0;
        assert!(invalid.validate().is_err());
        invalid.catchment_budget_generalized_km = 2_001.0;
        assert!(invalid.validate().is_err());
        invalid.catchment_budget_generalized_km = 1.0;
        invalid.freshwater_access_limit_generalized_km = 2_001.0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn robust_local_grade_does_not_turn_one_valley_wall_into_an_exclusion() {
        let tess = tessellation(80);
        let mut elevation = vec![0.0; tess.num_cells()];
        elevation[tess.neighbors(0)[0]] = 100.0;
        let submerged = vec![false; tess.num_cells()];
        let robust =
            local_trimmed_mean_absolute_land_grade(&tess, &elevation, &submerged, 0).unwrap();
        assert_eq!(robust, 0.0);
    }

    #[test]
    fn bounded_catchment_is_area_weighted_and_reports_repair_provenance() {
        let tess = tessellation(80);
        let n = tess.num_cells();
        let elevation = vec![0.0; n];
        let submerged = vec![false; n];
        let living: Vec<f32> = (0..n)
            .map(|cell| 0.1 + 0.8 * cell as f32 / (n - 1) as f32)
            .collect();
        let mut repaired = vec![false; n];
        repaired[0] = true;
        let mut scratch = vec![f32::INFINITY; n];
        let catchment = bounded_catchment(
            &tess,
            &elevation,
            &submerged,
            &living,
            &repaired,
            tess.cell_areas_ref(),
            0,
            5_000.0,
            config(),
            &mut scratch,
        )
        .unwrap();
        assert!(catchment.cells.len() > 1);
        assert!(catchment.maximum_cost_generalized_km > 0.0);
        let expected_living: f64 = catchment
            .cells
            .iter()
            .map(|entry| {
                entry.access_weight as f64
                    * solid_angle_to_km2(tess.cell_areas_ref()[entry.cell]) as f64
                    * living[entry.cell] as f64
            })
            .sum();
        assert!((catchment.living_opportunity_km2 as f64 - expected_living).abs() < 1.0);
        assert!(catchment.drainage_repair_effective_area_fraction > 0.0);
        assert!(catchment.drainage_repair_effective_area_fraction < 1.0);
    }

    #[test]
    fn catchment_fringe_claim_only_removes_its_marginal_access_weight() {
        let first = [CatchmentCell {
            cell: 0,
            access_weight: 0.1,
            living_contribution_km2: 10.0,
        }];
        let second = [CatchmentCell {
            cell: 0,
            access_weight: 1.0,
            living_contribution_km2: 100.0,
        }];
        let mut claimed = [0.0f32];
        for entry in first {
            claimed[entry.cell] = claimed[entry.cell].max(entry.access_weight);
        }
        assert!((unclaimed_living_opportunity(&second, &claimed) - 90.0).abs() < 1e-6);
    }

    #[test]
    fn hard_freshwater_viability_reports_shortfall_without_coast_compensation() {
        let tess = tessellation(180);
        let n = tess.num_cells();
        let elevation = vec![0.0; n];
        let submerged = vec![false; n];
        let mut components = synthetic_components(vec![1.0; n]);
        components.freshwater_access_generalized_km.fill(None);
        components.freshwater_source.fill(false);
        components.freshwater_source_kind.fill(None);
        components.freshwater_access_generalized_km[0] = Some(0.0);
        components.freshwater_source[0] = true;
        components.freshwater_source_kind[0] = Some(FreshwaterSourceKind::SelectedRiver);
        components.coast_access_generalized_km[1] = Some(0.0);
        components.coast_source[1] = true;

        let result = select_sites_from_fields(
            &tess,
            &elevation,
            &submerged,
            &components,
            site_config().validate().unwrap(),
        )
        .unwrap();
        assert_eq!(result.eligible_cell_count, 1);
        assert_eq!(result.sites.len(), 1);
        assert_eq!(result.site_shortfall, 11);
        assert_eq!(
            result.stop_reason,
            SiteSelectionStopReason::CandidateOrSpacingExhausted
        );
        assert_eq!(result.sites[0].anchor_cell, 0);
        assert_eq!(result.sites[0].nearest_freshwater_source_cell, 0);
        assert!(!result.sites.iter().any(|site| site.anchor_cell == 1));
    }

    #[test]
    fn site_selection_is_deterministic_and_enforces_physical_spacing() {
        let tess = tessellation(900);
        let n = tess.num_cells();
        let elevation = vec![0.0; n];
        let submerged = vec![false; n];
        let living: Vec<f32> = (0..n)
            .map(|cell| {
                let center = tess.cell_center(cell);
                let pattern =
                    (13.0 * center.x).sin() * (11.0 * center.y).sin() * (17.0 * center.z).sin();
                0.2 + 0.8 * (0.5 + 0.5 * pattern)
            })
            .collect();
        let components = synthetic_components(living);
        let mut selection_config = site_config();
        selection_config.candidate_pool_size = 80;
        selection_config.minimum_site_spacing_km = 800.0;
        selection_config.candidate_spacing_km = 200.0;
        selection_config.catchment_budget_generalized_km = 1_200.0;
        selection_config.coast_bonus = 0.0;
        let selection_config = selection_config.validate().unwrap();

        let first =
            select_sites_from_fields(&tess, &elevation, &submerged, &components, selection_config)
                .unwrap();
        let second =
            select_sites_from_fields(&tess, &elevation, &submerged, &components, selection_config)
                .unwrap();
        let first_cells: Vec<usize> = first.sites.iter().map(|site| site.anchor_cell).collect();
        let second_cells: Vec<usize> = second.sites.iter().map(|site| site.anchor_cell).collect();
        assert_eq!(first_cells, second_cells);
        assert_eq!(first.sites.len(), 12);
        assert!(first
            .sites
            .iter()
            .skip(1)
            .any(|site| site.claimed_opportunity_fraction > 0.0));
        for (index, site) in first.sites.iter().enumerate() {
            for other in first.sites.iter().skip(index + 1) {
                assert!(
                    physical_distance_km(&tess, site.anchor_cell, other.anchor_cell)
                        >= selection_config.minimum_site_spacing_km
                );
            }
        }
    }

    #[test]
    fn freshwater_and_coast_sources_keep_distinct_semantics() {
        let tess = tessellation(80);
        let n = tess.num_cells();
        let land = 0;
        let water_cell = tess.neighbors(land)[0];
        let mut submerged = vec![false; n];
        submerged[water_cell] = true;
        let no_rivers = vec![false; n];

        let ocean = one_cell_water(n, water_cell, SemanticWaterKind::Ocean);
        let (freshwater, kind, coast) = source_masks(&tess, &submerged, &ocean, &no_rivers);
        assert!(!freshwater[land]);
        assert_eq!(kind[land], None);
        assert!(coast[land]);

        let lake = one_cell_water(n, water_cell, SemanticWaterKind::Lake);
        let (freshwater, kind, coast) = source_masks(&tess, &submerged, &lake, &no_rivers);
        assert!(freshwater[land]);
        assert_eq!(kind[land], Some(FreshwaterSourceKind::ProperLakeShore));
        assert!(!coast[land]);

        let pond = one_cell_water(n, water_cell, SemanticWaterKind::Pond);
        let (freshwater, kind, coast) = source_masks(&tess, &submerged, &pond, &no_rivers);
        assert!(!freshwater[land]);
        assert_eq!(kind[land], None);
        assert!(!coast[land]);

        let empty_water = WaterBodySemantics {
            bodies: Vec::new(),
            cell_body: vec![None; n],
        };
        let mut river = no_rivers;
        river[land] = true;
        river[water_cell] = true;
        let (freshwater, kind, _) = source_masks(&tess, &submerged, &empty_water, &river);
        assert!(freshwater[land]);
        assert_eq!(kind[land], Some(FreshwaterSourceKind::SelectedRiver));
        assert!(!freshwater[water_cell]);
    }

    #[test]
    fn generated_world_builds_components_and_site_explanations_on_demand() {
        let tess = tessellation(600);
        let n = tess.num_cells();
        let elevation = Elevation {
            values: (0..n).map(|cell| 0.2 * tess.cell_center(cell).y).collect(),
            noise_contribution: vec![0.0; n],
            noise_layers: NoiseLayerData {
                macro_layer: vec![0.0; n],
            },
        };
        let temperature = vec![0.5; n];
        let precipitation = vec![1.0; n];
        let hydrology = Hydrology::generate_from_continentality(
            &tess,
            &vec![0.0; n],
            &elevation,
            &precipitation,
            &temperature,
        );
        let water = WaterBodySemantics::build(&tess, &hydrology);
        let rivers = RiverSelection::build(&hydrology, RiverThresholdPolicy::default());
        let living = LivingSurfaceSemantics::build(&tess, &temperature, &precipitation, &hydrology);
        let components = ConsequentialGeographyComponents::build(
            &tess,
            &hydrology,
            &water,
            &rivers,
            &living,
            config(),
        )
        .unwrap();
        let mut authored = site_config();
        authored.minimum_site_spacing_km = 100.0;
        authored.candidate_spacing_km = 50.0;
        authored.catchment_budget_generalized_km = 1_000.0;
        authored.freshwater_access_limit_generalized_km = 2_000.0;
        authored.minimum_local_living_opportunity = 0.0;
        authored.maximum_local_trimmed_mean_grade = 1.0;
        let selection = components
            .select_sites(&tess, &hydrology, authored)
            .unwrap();
        assert!(!selection.sites.is_empty());
        assert!(selection.sites.len() <= authored.site_count);
        assert_eq!(
            selection.site_shortfall,
            authored.site_count - selection.sites.len()
        );
        assert_eq!(selection.aggregate_river_policy, rivers.policy);
        assert!(selection.sites.iter().all(|site| {
            components.freshwater_source_kind[site.nearest_freshwater_source_cell]
                == Some(site.nearest_freshwater_source_kind)
        }));
    }

    #[test]
    fn flat_edges_cost_distance_and_reverse_slope_components_swap() {
        let tess = tessellation(80);
        let a = 0;
        let b = tess.neighbors(a)[0];
        let mut elevation = vec![0.0; tess.num_cells()];
        let flat = edge_cost_from_elevation(&tess, &elevation, a, b, false, config()).unwrap();
        assert_eq!(flat.generalized_cost_km, flat.distance_km);

        elevation[b] = 0.2;
        let up = edge_cost_from_elevation(&tess, &elevation, a, b, false, config()).unwrap();
        let down = edge_cost_from_elevation(&tess, &elevation, b, a, false, config()).unwrap();
        assert_eq!(up.distance_km.to_bits(), down.distance_km.to_bits());
        assert_eq!(up.ascent_km.to_bits(), down.descent_km.to_bits());
        assert_eq!(up.descent_km.to_bits(), down.ascent_km.to_bits());
        assert!(up.generalized_cost_km > down.generalized_cost_km);
    }

    #[test]
    fn neutral_access_is_physical_graph_distance_and_water_is_not_a_shortcut() {
        let tess = tessellation(100);
        let n = tess.num_cells();
        let elevation = vec![0.0; n];
        let mut sources = vec![false; n];
        sources[0] = true;
        let dry = vec![false; n];
        let access = access_costs(&tess, &elevation, &dry, &sources, config()).unwrap();
        assert_eq!(access[0], Some(0.0));
        assert!(access.iter().all(Option::is_some));

        let mut submerged = vec![true; n];
        submerged[0] = false;
        let isolated = access_costs(&tess, &elevation, &submerged, &sources, config()).unwrap();
        assert_eq!(isolated[0], Some(0.0));
        assert!(isolated.iter().skip(1).all(Option::is_none));
    }

    #[test]
    fn lower_gap_reduces_access_cost_across_a_steep_global_barrier() {
        let tess = tessellation(900);
        let n = tess.num_cells();
        let mut source = 0;
        let mut target = 0;
        for cell in 1..n {
            if tess.cell_center(cell).x < tess.cell_center(source).x {
                source = cell;
            }
            if tess.cell_center(cell).x > tess.cell_center(target).x {
                target = cell;
            }
        }
        let mut sources = vec![false; n];
        sources[source] = true;
        let submerged = vec![false; n];
        let mut closed = vec![0.0; n];
        for (cell, value) in closed.iter_mut().enumerate() {
            if tess.cell_center(cell).x.abs() < 0.28 {
                *value = 100.0;
            }
        }
        let mut open = closed.clone();
        for (cell, value) in open.iter_mut().enumerate() {
            let center = tess.cell_center(cell);
            if center.x.abs() < 0.28 && center.z > 0.72 {
                *value = 0.0;
            }
        }
        let closed_cost = access_costs(&tess, &closed, &submerged, &sources, config()).unwrap();
        let open_cost = access_costs(&tess, &open, &submerged, &sources, config()).unwrap();
        assert!(open_cost[target].unwrap() < closed_cost[target].unwrap());
    }
}
