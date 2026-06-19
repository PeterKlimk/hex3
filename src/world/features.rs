//! Tectonic feature fields derived from plate boundaries.
//!
//! This module computes canonical per-cell fields (trench, arc, ridge, collision, activity, regime)
//! from plate boundary edges. These fields are resolution-independent (distances in radians)
//! and serve as the primary drivers of terrain elevation.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::f32::consts::PI;

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use super::boundary::{collect_plate_boundaries, BoundaryKind, SubductionPolarity};
use super::constants::*;
use super::crust::{Crust, CrustType};
use super::dynamics::Dynamics;
use super::{Plates, Tessellation};

/// Tectonic feature fields derived from plate boundaries.
///
/// All values are resolution-independent magnitudes (not raw elevations).
/// Elevation generation applies these via decay functions and sensitivity constants.
pub struct FeatureFields {
    /// Signed trench dynamic-topography field.
    /// Positive values are downward deflection; negative values are outer-rise uplift.
    pub trench: Vec<f32>,

    /// Volcanic arc uplift (overriding side of subduction).
    /// Stores the computed uplift magnitude.
    pub arc: Vec<f32>,

    /// Mid-ocean ridge uplift (ocean-ocean divergent).
    /// Stores the computed uplift magnitude.
    pub ridge: Vec<f32>,

    /// Continental collision uplift (cont-cont convergent).
    /// Stores the computed uplift magnitude.
    pub collision: Vec<f32>,

    /// Tectonic activity scalar (0-1).
    /// High near active boundaries, decays into plate interiors.
    /// Used for noise modulation and roughness.
    pub activity: Vec<f32>,

    /// Convergent boundary influence scalar (0-1).
    /// High near convergent boundaries, decays into plate interiors.
    /// Used for regime-aware noise modulation (compressional texture).
    pub convergent: Vec<f32>,

    /// Divergent boundary influence scalar (0-1).
    /// High near divergent boundaries, decays into plate interiors.
    /// Used for regime-aware noise modulation (extensional texture).
    pub divergent: Vec<f32>,

    /// Transform boundary influence scalar (0-1).
    /// High near transform boundaries, decays into plate interiors.
    /// Reserved for future directional/shear-aware noise.
    pub transform: Vec<f32>,

    /// Raw distance from nearest mid-ocean ridge (radians).
    /// Stored for diagnostics/visualization; age-sensitive consumers use
    /// `ridge_age_distance`.
    /// Infinity for cells with no ridge on their plate.
    pub ridge_distance: Vec<f32>,

    /// Distance-from-ridge converted to age-equivalent distance using local
    /// spreading rate. Matches `ridge_distance` at the reference opening rate.
    pub ridge_age_distance: Vec<f32>,

    /// Opening rate propagated from the nearest mid-ocean ridge segment.
    pub ridge_spreading_rate: Vec<f32>,

    /// Raw distance from nearest continental collision boundary (radians).
    /// Infinity for cells with no collision boundary on their plate.
    pub collision_distance: Vec<f32>,

    /// Raw distance from nearest volcanic arc boundary (radians).
    /// Combines both continental and oceanic arcs.
    /// Infinity for cells with no arc boundary on their plate.
    pub arc_distance: Vec<f32>,

    /// Per-cell arc shape noise used for oceanic arc coastline variation.
    /// Stored for visualization; applied additively to arc uplift.
    pub arc_shape_noise: Vec<f32>,

    /// Signed crustal thickness change from continental rifting (thickness
    /// units): negative in the axial valley (necking-localized thinning
    /// driven by the boundary opening rate), positive on the uplifted
    /// shoulders. Consumed by elevation through isostasy.
    pub rift_delta: Vec<f32>,
}

impl FeatureFields {
    /// Compute all feature fields from plate boundaries.
    pub fn compute(
        tessellation: &Tessellation,
        plates: &Plates,
        crust: &Crust,
        dynamics: &Dynamics,
    ) -> Self {
        let boundaries = collect_plate_boundaries(tessellation, plates, crust, dynamics);
        let num_cells = tessellation.num_cells();
        let boundary_edge_midpoints = build_cell_pair_edge_midpoints(tessellation);

        // Cell areas for resolution-independent forcing normalization.
        // Forces are scaled by mean_area/cell_area so that total integrated
        // forcing is constant regardless of resolution.
        let cell_areas = tessellation.cell_areas();
        let mean_area = tessellation.mean_cell_area();
        let area_scale = |cell_idx: usize| -> f32 { mean_area / cell_areas[cell_idx].max(1e-10) };

        // Mean neighbor distance for adaptive smoothing iterations.
        // Smoothing spreads ~1 neighbor hop per iteration, so we need more
        // iterations at higher resolutions to cover the same physical distance.
        let mean_neighbor_dist = compute_mean_neighbor_distance(tessellation);

        // Build edge-anchored seed arrays for each feature type.
        //
        // We keep per-cell seed strengths for summing contributions, but the "source distance"
        // is the distance from the cell center to the shared Voronoi edge midpoint. This avoids
        // quantizing all features to "0 at boundary cell centers", which makes arcs/trenches
        // appear too close to boundaries at coarse resolutions.
        let mut trench_seed_strength = vec![0.0f32; num_cells];
        let mut trench_seed_dist0 = vec![f32::INFINITY; num_cells];

        let mut forearc_seed_strength = vec![0.0f32; num_cells];
        let mut forearc_seed_dist0 = vec![f32::INFINITY; num_cells];

        let mut arc_seed_strength_cont = vec![0.0f32; num_cells];
        let mut arc_seed_dist0_cont = vec![f32::INFINITY; num_cells];

        let mut arc_seed_strength_ocean = vec![0.0f32; num_cells];
        let mut arc_seed_dist0_ocean = vec![f32::INFINITY; num_cells];

        let mut ridge_seed_strength_ocean = vec![0.0f32; num_cells];
        let mut ridge_seed_dist0_ocean = vec![f32::INFINITY; num_cells];
        let mut ridge_opening_seed_sum = vec![0.0f32; num_cells];
        let mut ridge_opening_seed_count = vec![0.0f32; num_cells];

        let mut collision_seed_strength = vec![0.0f32; num_cells];
        let mut collision_seed_dist0 = vec![f32::INFINITY; num_cells];

        let mut rift_seed_strength = vec![0.0f32; num_cells];
        let mut rift_seed_dist0 = vec![f32::INFINITY; num_cells];

        // Per-cell boundary length contributing to each magnitude-feature seed.
        // The seed is accumulated as Σ(rate·mult·SCALE·edge_length) and divided by
        // this Σ(edge_length) after the loop, yielding an edge-length-weighted MEAN
        // rate (intensive, resolution-invariant). See `normalize_force_seed` and
        // FEATURE_FORCE_REF_SPACING.
        let mut trench_seed_weight = vec![0.0f32; num_cells];
        let mut forearc_seed_weight = vec![0.0f32; num_cells];
        let mut arc_seed_weight_cont = vec![0.0f32; num_cells];
        let mut arc_seed_weight_ocean = vec![0.0f32; num_cells];
        let mut ridge_seed_weight_ocean = vec![0.0f32; num_cells];
        let mut collision_seed_weight = vec![0.0f32; num_cells];
        let mut rift_seed_weight = vec![0.0f32; num_cells];

        let mut activity_seed = vec![0.0f32; num_cells];
        let mut convergent_seed = vec![0.0f32; num_cells];
        let mut divergent_seed = vec![0.0f32; num_cells];
        let mut transform_seed = vec![0.0f32; num_cells];

        for b in &boundaries {
            // Activity: all boundary cells get activity based on relative speed
            // Normalize by cell area for resolution-independent forcing
            let activity_force = b.relative_speed * b.edge_length;
            activity_seed[b.cell_a] += activity_force * area_scale(b.cell_a);
            activity_seed[b.cell_b] += activity_force * area_scale(b.cell_b);

            // Regime influence: split into convergent/divergent/transform boundary drivers.
            // These are magnitude-only kinematic weights used for noise modulation, not
            // for the feature magnitudes (those use `closing` etc below).
            match b.kind {
                BoundaryKind::Convergent => {
                    let closing = b.convergence.max(0.0);
                    let force = closing * b.edge_length;
                    convergent_seed[b.cell_a] += force * area_scale(b.cell_a);
                    convergent_seed[b.cell_b] += force * area_scale(b.cell_b);
                }
                BoundaryKind::Divergent => {
                    let opening = (-b.convergence).max(0.0);
                    let force = opening * b.edge_length;
                    divergent_seed[b.cell_a] += force * area_scale(b.cell_a);
                    divergent_seed[b.cell_b] += force * area_scale(b.cell_b);
                }
                BoundaryKind::Transform => {
                    let shear = b.shear.abs();
                    let force = shear * b.edge_length;
                    transform_seed[b.cell_a] += force * area_scale(b.cell_a);
                    transform_seed[b.cell_b] += force * area_scale(b.cell_b);
                }
            }

            let edge_midpoint = cell_pair_edge_midpoint(
                tessellation,
                &boundary_edge_midpoints,
                b.cell_a,
                b.cell_b,
                b.boundary_point,
            );
            let dist0_a = angular_distance(tessellation.cell_center(b.cell_a), edge_midpoint);
            let dist0_b = angular_distance(tessellation.cell_center(b.cell_b), edge_midpoint);

            match b.kind {
                BoundaryKind::Convergent => {
                    // Only use closing motion; avoid "convergent regime" edges that are locally
                    // divergent due to boundary geometry noise.
                    let closing = b.convergence.max(0.0);
                    if closing < TRANSFORM_NORMAL_THRESHOLD {
                        continue;
                    }

                    // Compute per-side forcing
                    //
                    // Note: uplift-style features (arcs/collisions) and trench depth are scaled
                    // separately so that trench depth isn't accidentally suppressed on the
                    // subducting oceanic side.
                    let uplift_mult_a = uplift_multiplier(b.type_a, b.type_b);
                    let uplift_mult_b = uplift_multiplier(b.type_b, b.type_a);
                    let uplift_force_a =
                        closing * uplift_mult_a * b.edge_length * FEATURE_FORCE_SCALE;
                    let uplift_force_b =
                        closing * uplift_mult_b * b.edge_length * FEATURE_FORCE_SCALE;

                    let subd_mult_a = subduction_multiplier(b.type_a, b.type_b);
                    let subd_mult_b = subduction_multiplier(b.type_b, b.type_a);
                    let subd_force_a = closing * subd_mult_a * b.edge_length * FEATURE_FORCE_SCALE;
                    let subd_force_b = closing * subd_mult_b * b.edge_length * FEATURE_FORCE_SCALE;

                    // Handle subduction (trench + arc) vs collision.
                    //
                    // Seeds accumulate Σ(force·edge_length already folded in) and a
                    // parallel Σ(edge_length) weight; `normalize_force_seed` divides
                    // them after the loop to an intensive (resolution-invariant) mean.
                    // `area_scale` is intentionally gone — the edge-length-weighted
                    // mean already handles per-cell normalization without the cell
                    // count dependence area_scale's sum form carried.
                    if let Some(polarity) = b.subduction {
                        match polarity {
                            SubductionPolarity::ASubducts => {
                                // A subducts: trench on A if oceanic; arc on B (overriding)
                                if b.type_a == CrustType::Oceanic {
                                    add_force_seed(
                                        &mut trench_seed_strength,
                                        &mut trench_seed_weight,
                                        b.cell_a,
                                        subd_force_a,
                                        b.edge_length,
                                    );
                                    trench_seed_dist0[b.cell_a] =
                                        trench_seed_dist0[b.cell_a].min(dist0_a);
                                    add_force_seed(
                                        &mut forearc_seed_strength,
                                        &mut forearc_seed_weight,
                                        b.cell_b,
                                        subd_force_a,
                                        b.edge_length,
                                    );
                                    forearc_seed_dist0[b.cell_b] =
                                        forearc_seed_dist0[b.cell_b].min(dist0_b);
                                }
                                match b.type_b {
                                    CrustType::Continental => {
                                        add_force_seed(
                                            &mut arc_seed_strength_cont,
                                            &mut arc_seed_weight_cont,
                                            b.cell_b,
                                            uplift_force_b,
                                            b.edge_length,
                                        );
                                        arc_seed_dist0_cont[b.cell_b] =
                                            arc_seed_dist0_cont[b.cell_b].min(dist0_b);
                                    }
                                    CrustType::Oceanic => {
                                        add_force_seed(
                                            &mut arc_seed_strength_ocean,
                                            &mut arc_seed_weight_ocean,
                                            b.cell_b,
                                            uplift_force_b,
                                            b.edge_length,
                                        );
                                        arc_seed_dist0_ocean[b.cell_b] =
                                            arc_seed_dist0_ocean[b.cell_b].min(dist0_b);
                                    }
                                }
                            }
                            SubductionPolarity::BSubducts => {
                                if b.type_b == CrustType::Oceanic {
                                    add_force_seed(
                                        &mut trench_seed_strength,
                                        &mut trench_seed_weight,
                                        b.cell_b,
                                        subd_force_b,
                                        b.edge_length,
                                    );
                                    trench_seed_dist0[b.cell_b] =
                                        trench_seed_dist0[b.cell_b].min(dist0_b);
                                    add_force_seed(
                                        &mut forearc_seed_strength,
                                        &mut forearc_seed_weight,
                                        b.cell_a,
                                        subd_force_b,
                                        b.edge_length,
                                    );
                                    forearc_seed_dist0[b.cell_a] =
                                        forearc_seed_dist0[b.cell_a].min(dist0_a);
                                }
                                match b.type_a {
                                    CrustType::Continental => {
                                        add_force_seed(
                                            &mut arc_seed_strength_cont,
                                            &mut arc_seed_weight_cont,
                                            b.cell_a,
                                            uplift_force_a,
                                            b.edge_length,
                                        );
                                        arc_seed_dist0_cont[b.cell_a] =
                                            arc_seed_dist0_cont[b.cell_a].min(dist0_a);
                                    }
                                    CrustType::Oceanic => {
                                        add_force_seed(
                                            &mut arc_seed_strength_ocean,
                                            &mut arc_seed_weight_ocean,
                                            b.cell_a,
                                            uplift_force_a,
                                            b.edge_length,
                                        );
                                        arc_seed_dist0_ocean[b.cell_a] =
                                            arc_seed_dist0_ocean[b.cell_a].min(dist0_a);
                                    }
                                }
                            }
                        }
                    } else {
                        // No subduction polarity = continent-continent collision
                        if b.type_a == CrustType::Continental && b.type_b == CrustType::Continental
                        {
                            add_force_seed(
                                &mut collision_seed_strength,
                                &mut collision_seed_weight,
                                b.cell_a,
                                uplift_force_a,
                                b.edge_length,
                            );
                            collision_seed_dist0[b.cell_a] =
                                collision_seed_dist0[b.cell_a].min(dist0_a);
                            add_force_seed(
                                &mut collision_seed_strength,
                                &mut collision_seed_weight,
                                b.cell_b,
                                uplift_force_b,
                                b.edge_length,
                            );
                            collision_seed_dist0[b.cell_b] =
                                collision_seed_dist0[b.cell_b].min(dist0_b);
                        }
                    }
                }
                BoundaryKind::Divergent => {
                    // Only use opening motion; avoid "divergent regime" edges that are locally
                    // convergent due to boundary geometry noise.
                    let opening = (-b.convergence).max(0.0);
                    if opening < TRANSFORM_NORMAL_THRESHOLD {
                        continue;
                    }

                    // Mid-ocean ridges for ocean-ocean divergence
                    if b.type_a == CrustType::Oceanic && b.type_b == CrustType::Oceanic {
                        let force = opening * DIV_OCEAN_OCEAN * b.edge_length * FEATURE_FORCE_SCALE;
                        add_force_seed(
                            &mut ridge_seed_strength_ocean,
                            &mut ridge_seed_weight_ocean,
                            b.cell_a,
                            force,
                            b.edge_length,
                        );
                        ridge_seed_dist0_ocean[b.cell_a] =
                            ridge_seed_dist0_ocean[b.cell_a].min(dist0_a);
                        ridge_opening_seed_sum[b.cell_a] += opening;
                        ridge_opening_seed_count[b.cell_a] += 1.0;
                        add_force_seed(
                            &mut ridge_seed_strength_ocean,
                            &mut ridge_seed_weight_ocean,
                            b.cell_b,
                            force,
                            b.edge_length,
                        );
                        ridge_seed_dist0_ocean[b.cell_b] =
                            ridge_seed_dist0_ocean[b.cell_b].min(dist0_b);
                        ridge_opening_seed_sum[b.cell_b] += opening;
                        ridge_opening_seed_count[b.cell_b] += 1.0;
                    }

                    // Continental rifting: seed thinning on continental cells
                    // from the actual per-edge opening rate (kinematic, so
                    // along-strike variation comes from Euler-pole geometry,
                    // not a normalized influence field).
                    if b.type_a == CrustType::Continental {
                        let mult = if b.type_b == CrustType::Continental {
                            DIV_CONT_CONT
                        } else {
                            DIV_CONT_OCEAN
                        };
                        let force = opening * mult * b.edge_length * FEATURE_FORCE_SCALE;
                        add_force_seed(
                            &mut rift_seed_strength,
                            &mut rift_seed_weight,
                            b.cell_a,
                            force,
                            b.edge_length,
                        );
                        rift_seed_dist0[b.cell_a] = rift_seed_dist0[b.cell_a].min(dist0_a);
                    }
                    if b.type_b == CrustType::Continental {
                        let mult = if b.type_a == CrustType::Continental {
                            DIV_CONT_CONT
                        } else {
                            DIV_CONT_OCEAN
                        };
                        let force = opening * mult * b.edge_length * FEATURE_FORCE_SCALE;
                        add_force_seed(
                            &mut rift_seed_strength,
                            &mut rift_seed_weight,
                            b.cell_b,
                            force,
                            b.edge_length,
                        );
                        rift_seed_dist0[b.cell_b] = rift_seed_dist0[b.cell_b].min(dist0_b);
                    }
                }
                BoundaryKind::Transform => {
                    // Transforms don't produce elevation features
                    // (activity already captured above)
                }
            }
        }

        // Convert the accumulated Σ(force·edge_length) seeds into intensive,
        // resolution-invariant per-cell forcing (edge-length-weighted mean rate ×
        // FEATURE_FORCE_REF_SPACING). Distance fields below only test strength > 0
        // (presence), so this normalization leaves the geometry unchanged; it
        // rescales the AMPLITUDES that feed `compute_smoothed_boundary_forcing` ->
        // `sqrt_response`, removing the old ~1/sqrt(N) cell-count dependence.
        normalize_force_seed(&mut trench_seed_strength, &trench_seed_weight);
        normalize_force_seed(&mut forearc_seed_strength, &forearc_seed_weight);
        normalize_force_seed(&mut arc_seed_strength_cont, &arc_seed_weight_cont);
        normalize_force_seed(&mut arc_seed_strength_ocean, &arc_seed_weight_ocean);
        normalize_force_seed(&mut ridge_seed_strength_ocean, &ridge_seed_weight_ocean);
        normalize_force_seed(&mut collision_seed_strength, &collision_seed_weight);
        normalize_force_seed(&mut rift_seed_strength, &rift_seed_weight);

        // Compute edge-anchored distance fields from seeds.
        let trench_dist = distance_field_from_edge_seed_cells(
            tessellation,
            plates,
            &trench_seed_strength,
            &trench_seed_dist0,
            true,
        );
        let forearc_dist = distance_field_from_edge_seed_cells(
            tessellation,
            plates,
            &forearc_seed_strength,
            &forearc_seed_dist0,
            true,
        );
        let arc_dist_cont = distance_field_from_edge_seed_cells(
            tessellation,
            plates,
            &arc_seed_strength_cont,
            &arc_seed_dist0_cont,
            true,
        );
        let arc_dist_ocean = distance_field_from_edge_seed_cells(
            tessellation,
            plates,
            &arc_seed_strength_ocean,
            &arc_seed_dist0_ocean,
            true,
        );
        // Combined arc distance (min of continental and oceanic)
        let arc_dist: Vec<f32> = arc_dist_cont
            .iter()
            .zip(arc_dist_ocean.iter())
            .map(|(&c, &o)| c.min(o))
            .collect();
        let ridge_seed_opening_rate: Vec<f32> = ridge_opening_seed_sum
            .iter()
            .zip(ridge_opening_seed_count.iter())
            .map(|(&sum, &count)| if count > 0.0 { sum / count } else { 0.0 })
            .collect();
        let (ridge_dist, ridge_spreading_rate) = distance_and_value_field_from_edge_seed_cells(
            tessellation,
            plates,
            &ridge_seed_strength_ocean,
            &ridge_seed_dist0_ocean,
            &ridge_seed_opening_rate,
            true,
        );
        let ridge_age_dist: Vec<f32> = ridge_dist
            .iter()
            .zip(ridge_spreading_rate.iter())
            .map(|(&dist, &rate)| ridge_age_distance_from_spreading_rate(dist, rate))
            .collect();
        let rift_support_dist = RIFT_SHOULDER_OFFSET + 3.0 * RIFT_SHOULDER_WIDTH;
        let rift_dist = distance_field_from_edge_seed_cells(
            tessellation,
            plates,
            &rift_seed_strength,
            &rift_seed_dist0,
            true,
        );
        let collision_dist = distance_field_from_edge_seed_cells(
            tessellation,
            plates,
            &collision_seed_strength,
            &collision_seed_dist0,
            true,
        );

        // Compute smoothed boundary forcing fields (amplitudes).
        //
        // These are "normalized diffusions": we diffuse both the boundary forcing and a unit
        // weight field, and take their ratio. This smooths the forcing along/between nearby
        // boundary segments without introducing an additional inland decay (the distance kernels
        // below remain the primary inland projection).
        let trench_support_dist = (PI + 1.0) * TRENCH_FLEX_ALPHA * TRENCH_FLEX_ALPHA_OLD_MULT;
        let forearc_support_dist = (0.75 * PI + 1.0) * FOREARC_ALPHA;
        let arc_cont_support_dist = ARC_CONT_PEAK_DIST + 3.0 * ARC_CONT_WIDTH;
        let arc_ocean_support_dist = ARC_OCEAN_PEAK_DIST + 3.0 * ARC_OCEAN_WIDTH;
        let ridge_support_dist = 4.0 * RIDGE_DECAY;
        let collision_support_dist = COLLISION_PEAK_DIST + 3.0 * COLLISION_WIDTH;

        let trench_forcing = compute_smoothed_boundary_forcing(
            tessellation,
            plates,
            &trench_seed_strength,
            trench_support_dist,
            mean_neighbor_dist,
        );
        let forearc_forcing = compute_smoothed_boundary_forcing(
            tessellation,
            plates,
            &forearc_seed_strength,
            forearc_support_dist,
            mean_neighbor_dist,
        );
        let arc_forcing_cont = compute_smoothed_boundary_forcing(
            tessellation,
            plates,
            &arc_seed_strength_cont,
            arc_cont_support_dist,
            mean_neighbor_dist,
        );
        let arc_forcing_ocean = compute_smoothed_boundary_forcing(
            tessellation,
            plates,
            &arc_seed_strength_ocean,
            arc_ocean_support_dist,
            mean_neighbor_dist,
        );
        let ridge_forcing_ocean = compute_smoothed_boundary_forcing(
            tessellation,
            plates,
            &ridge_seed_strength_ocean,
            ridge_support_dist,
            mean_neighbor_dist,
        );
        let collision_forcing = compute_smoothed_boundary_forcing(
            tessellation,
            plates,
            &collision_seed_strength,
            collision_support_dist,
            mean_neighbor_dist,
        );
        let rift_forcing = compute_smoothed_boundary_forcing(
            tessellation,
            plates,
            &rift_seed_strength,
            rift_support_dist,
            mean_neighbor_dist,
        );

        // Compute activity and regime influence via diffusion (plate-constrained).
        let activity = compute_influence_field(
            tessellation,
            plates,
            &activity_seed,
            ACTIVITY_INFLUENCE_LENGTH,
            mean_neighbor_dist,
        );
        let convergent = compute_influence_field(
            tessellation,
            plates,
            &convergent_seed,
            CONVERGENT_INFLUENCE_LENGTH,
            mean_neighbor_dist,
        );
        let divergent = compute_influence_field(
            tessellation,
            plates,
            &divergent_seed,
            DIVERGENT_INFLUENCE_LENGTH,
            mean_neighbor_dist,
        );
        let transform = compute_influence_field(
            tessellation,
            plates,
            &transform_seed,
            TRANSFORM_INFLUENCE_LENGTH,
            mean_neighbor_dist,
        );

        // Convert distance fields to feature magnitudes
        let mut trench = vec![0.0f32; num_cells];
        let mut arc = vec![0.0f32; num_cells];
        let mut ridge = vec![0.0f32; num_cells];
        let mut collision = vec![0.0f32; num_cells];
        let mut arc_shape_noise = vec![0.0f32; num_cells];
        let mut rift_delta = vec![0.0f32; num_cells];

        // Additive noise for oceanic arc height variation.
        let arc_noise_fbm: Fbm<Perlin> = Fbm::new(ARC_NOISE_SEED).set_octaves(ARC_NOISE_OCTAVES);

        for i in 0..num_cells {
            let crust_type = crust.crust_type(i);
            let is_continental = crust_type == CrustType::Continental;

            // Trench: oceanic only
            if crust_type == CrustType::Oceanic {
                let d = trench_dist[i];
                if d.is_finite() {
                    // Slab age modulation: older oceanic lithosphere tends to produce stronger
                    // trench/slab-pull signals than very young crust near ridges.
                    let age = oceanic_age_factor_from_ridge_distance(ridge_age_dist[i]);
                    let age_mult =
                        TRENCH_AGE_YOUNG_MULT + (TRENCH_AGE_OLD_MULT - TRENCH_AGE_YOUNG_MULT) * age;

                    let depth = sqrt_response(
                        trench_forcing[i] * age_mult,
                        TRENCH_SENSITIVITY,
                        TRENCH_MAX_DEPTH,
                    );
                    let alpha = TRENCH_FLEX_ALPHA
                        * (TRENCH_FLEX_ALPHA_YOUNG_MULT
                            + (TRENCH_FLEX_ALPHA_OLD_MULT - TRENCH_FLEX_ALPHA_YOUNG_MULT) * age);
                    trench[i] = depth * flexure_broken(d, alpha);
                }
            }

            if forearc_dist[i].is_finite() {
                let w0 = FOREARC_COUPLING
                    * sqrt_response(forearc_forcing[i], TRENCH_SENSITIVITY, TRENCH_MAX_DEPTH);
                trench[i] += w0 * flexure_coupled(forearc_dist[i], FOREARC_ALPHA);
            }

            // Arc uplift: continental or oceanic depending on crust type
            let (arc_dist_val, forcing) = if is_continental {
                (arc_dist_cont[i], arc_forcing_cont[i])
            } else {
                (arc_dist_ocean[i], arc_forcing_ocean[i])
            };
            if arc_dist_val.is_finite() {
                let (sens, max_uplift, peak, width) = if is_continental {
                    (
                        ARC_CONT_SENSITIVITY,
                        ARC_CONT_MAX_UPLIFT,
                        ARC_CONT_PEAK_DIST,
                        ARC_CONT_WIDTH,
                    )
                } else {
                    (
                        ARC_OCEAN_SENSITIVITY,
                        ARC_OCEAN_MAX_UPLIFT,
                        ARC_OCEAN_PEAK_DIST,
                        ARC_OCEAN_WIDTH,
                    )
                };
                let uplift = sqrt_response(forcing, sens, max_uplift);
                let mut val = uplift * gaussian_band(arc_dist_val, peak, width);

                // Oceanic arcs: multiplicative noise to create island clustering.
                // Noise determines which parts of the arc form islands vs remain underwater.
                if !is_continental && val > 0.0 {
                    let pos = tessellation.cell_center(i);
                    let p = pos * ARC_NOISE_FREQ as f32;
                    let noise_sample =
                        arc_noise_fbm.get([p.x as f64, p.y as f64, p.z as f64]) as f32;

                    // Convert noise to 0-1 modulation using smoothstep around threshold
                    let modulation = smoothstep(
                        ARC_ISLAND_THRESHOLD - ARC_ISLAND_TRANSITION,
                        ARC_ISLAND_THRESHOLD + ARC_ISLAND_TRANSITION,
                        noise_sample,
                    );
                    val *= modulation;

                    // Store for visualization
                    arc_shape_noise[i] = noise_sample;
                }

                arc[i] = val.max(0.0);
            }

            // Ridge: oceanic only
            if crust_type == CrustType::Oceanic {
                let d = ridge_dist[i];
                if d.is_finite() {
                    let uplift =
                        sqrt_response(ridge_forcing_ocean[i], RIDGE_SENSITIVITY, RIDGE_MAX_UPLIFT);
                    ridge[i] = uplift * exp_decay(d, RIDGE_DECAY);
                }
            }

            // Collision: continental only
            if is_continental {
                let d = collision_dist[i];
                if d.is_finite() {
                    let uplift = sqrt_response(
                        collision_forcing[i],
                        COLLISION_SENSITIVITY,
                        COLLISION_MAX_UPLIFT,
                    );
                    collision[i] = uplift * gaussian_band(d, COLLISION_PEAK_DIST, COLLISION_WIDTH);
                }
            }

            // Continental rift: necking-localized axial thinning with
            // flexural shoulder uplift on the flanks.
            if is_continental {
                let d = rift_dist[i];
                if d.is_finite() {
                    let thinning =
                        sqrt_response(rift_forcing[i], RIFT_SENSITIVITY, RIFT_MAX_THINNING);
                    let axial = gaussian_band(d, 0.0, RIFT_VALLEY_WIDTH);
                    let shoulder = RIFT_SHOULDER_RATIO
                        * gaussian_band(d, RIFT_SHOULDER_OFFSET, RIFT_SHOULDER_WIDTH);
                    rift_delta[i] = thinning * (shoulder - axial);
                }
            }
        }

        // Diagnostic logging for resolution-independence verification.
        // Values should be similar regardless of cell count.
        if log::log_enabled!(log::Level::Debug) {
            let trench_sum: f32 = trench.iter().sum();
            let trench_max = trench.iter().cloned().fold(0.0f32, f32::max);
            let arc_sum: f32 = arc.iter().sum();
            let arc_max = arc.iter().cloned().fold(0.0f32, f32::max);
            let ridge_sum: f32 = ridge.iter().sum();
            let ridge_max = ridge.iter().cloned().fold(0.0f32, f32::max);
            let collision_sum: f32 = collision.iter().sum();
            let collision_max = collision.iter().cloned().fold(0.0f32, f32::max);
            let activity_sum: f32 = activity.iter().sum();
            let activity_max = activity.iter().cloned().fold(0.0f32, f32::max);

            log::debug!(
                "Features @ {} cells: mean_dist={:.4}, mean_area={:.6}",
                num_cells,
                mean_neighbor_dist,
                mean_area
            );
            log::debug!(
                "  Trench: sum={:.2}, max={:.3} | Arc: sum={:.2}, max={:.3}",
                trench_sum,
                trench_max,
                arc_sum,
                arc_max
            );
            log::debug!(
                "  Ridge: sum={:.2}, max={:.3} | Collision: sum={:.2}, max={:.3}",
                ridge_sum,
                ridge_max,
                collision_sum,
                collision_max
            );
            log::debug!(
                "  Activity: sum={:.2}, max={:.3} | Convergent max={:.3} | Divergent max={:.3} | Transform max={:.3}",
                activity_sum,
                activity_max,
                convergent.iter().cloned().fold(0.0f32, f32::max),
                divergent.iter().cloned().fold(0.0f32, f32::max),
                transform.iter().cloned().fold(0.0f32, f32::max),
            );
        }

        Self {
            trench,
            arc,
            ridge,
            collision,
            rift_delta,
            activity,
            convergent,
            divergent,
            transform,
            ridge_distance: ridge_dist,
            ridge_age_distance: ridge_age_dist,
            ridge_spreading_rate,
            collision_distance: collision_dist,
            arc_distance: arc_dist,
            arc_shape_noise,
        }
    }
}

fn distance_and_value_field_from_edge_seed_cells(
    tessellation: &Tessellation,
    plates: &Plates,
    seed_strength: &[f32],
    seed_dist0: &[f32],
    seed_value: &[f32],
    restrict_to_plate: bool,
) -> (Vec<f32>, Vec<f32>) {
    #[derive(Clone, Copy, PartialEq)]
    struct State {
        dist: f32,
        cell: usize,
        plate: u32,
        value: f32,
    }

    impl Eq for State {}

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .dist
                .partial_cmp(&self.dist)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = tessellation.num_cells();
    let mut dist = vec![f32::INFINITY; n];
    let mut value = vec![0.0f32; n];
    let mut heap = BinaryHeap::new();

    for i in 0..n {
        let d0 = seed_dist0[i];
        if seed_strength[i] > 0.0 && d0.is_finite() {
            let plate = plates.cell_plate[i];
            dist[i] = dist[i].min(d0);
            value[i] = seed_value[i];
            heap.push(State {
                dist: d0,
                cell: i,
                plate,
                value: seed_value[i],
            });
        }
    }

    const TIE_EPS: f32 = 1e-6;

    while let Some(State {
        dist: d,
        cell,
        plate,
        value: source_value,
    }) = heap.pop()
    {
        if d > dist[cell] + TIE_EPS {
            continue;
        }

        let pos = tessellation.cell_center(cell);

        for &neighbor in tessellation.neighbors(cell) {
            if restrict_to_plate && plates.cell_plate[neighbor] != plate {
                continue;
            }

            let neighbor_pos = tessellation.cell_center(neighbor);
            let step = angular_distance(pos, neighbor_pos);
            let nd = d + step;

            if nd + TIE_EPS < dist[neighbor] {
                dist[neighbor] = nd;
                value[neighbor] = source_value;
                heap.push(State {
                    dist: nd,
                    cell: neighbor,
                    plate,
                    value: source_value,
                });
            }
        }
    }

    (dist, value)
}

fn angular_distance(a: Vec3, b: Vec3) -> f32 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

pub(crate) fn build_cell_pair_edge_midpoints(
    tessellation: &Tessellation,
) -> HashMap<(usize, usize), Vec3> {
    // Map Voronoi edges (vertex pairs) -> cells containing that edge, then produce a
    // cell-pair map keyed by (min_cell, max_cell) to the edge midpoint.
    let voronoi = &tessellation.voronoi;
    let mut edge_to_cells: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let verts = cell.vertex_indices;
        let n = verts.len();
        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }
    }

    let mut cell_pair_to_midpoint: HashMap<(usize, usize), Vec3> = HashMap::new();

    for ((va, vb), cells) in edge_to_cells {
        if cells.len() != 2 {
            continue;
        }
        let c0 = cells[0];
        let c1 = cells[1];
        let key = if c0 < c1 { (c0, c1) } else { (c1, c0) };

        let v0 = voronoi.vertices[va as usize];
        let v1 = voronoi.vertices[vb as usize];
        let sum = v0 + v1;
        let midpoint = if sum.length_squared() > 1e-10 {
            sum.normalize()
        } else {
            v0
        };

        cell_pair_to_midpoint.insert(key, midpoint);
    }

    cell_pair_to_midpoint
}

/// Like [`build_cell_pair_edge_midpoints`] but returns the shared Voronoi edge's two
/// VERTEX endpoints (unit vectors) per cell pair, so a consumer can measure distance
/// to the boundary as a great-circle ARC rather than to a single anchor point (whose
/// iso-distance contours scallop into bullseyes). Used by the P1b strike-aware relief.
pub(crate) fn build_cell_pair_edge_endpoints(
    tessellation: &Tessellation,
) -> HashMap<(usize, usize), (Vec3, Vec3)> {
    let voronoi = &tessellation.voronoi;
    let mut edge_to_cells: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let verts = cell.vertex_indices;
        let n = verts.len();
        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }
    }

    let mut cell_pair_to_endpoints: HashMap<(usize, usize), (Vec3, Vec3)> = HashMap::new();
    for ((va, vb), cells) in edge_to_cells {
        if cells.len() != 2 {
            continue;
        }
        let key = if cells[0] < cells[1] {
            (cells[0], cells[1])
        } else {
            (cells[1], cells[0])
        };
        let v0 = voronoi.vertices[va as usize];
        let v1 = voronoi.vertices[vb as usize];
        cell_pair_to_endpoints.insert(key, (v0, v1));
    }
    cell_pair_to_endpoints
}

fn cell_pair_edge_midpoint(
    tessellation: &Tessellation,
    midpoints: &HashMap<(usize, usize), Vec3>,
    cell_a: usize,
    cell_b: usize,
    fallback: Vec3,
) -> Vec3 {
    let key = if cell_a < cell_b {
        (cell_a, cell_b)
    } else {
        (cell_b, cell_a)
    };
    midpoints.get(&key).copied().unwrap_or_else(|| {
        // Fallback: use boundary midpoint between cell centers (still on the interface),
        // and normalize defensively.
        if fallback.length_squared() > 1e-10 {
            fallback.normalize()
        } else {
            tessellation.cell_center(cell_a)
        }
    })
}

/// Accumulate one boundary edge's contribution to a magnitude-feature seed:
/// `force` (which already folds in `edge_length`) into the strength sum, and
/// `edge_length` into the weight sum. After the boundary loop,
/// `normalize_force_seed` divides the two to an intensive mean.
#[inline]
fn add_force_seed(
    strength: &mut [f32],
    weight: &mut [f32],
    cell: usize,
    force: f32,
    edge_length: f32,
) {
    strength[cell] += force;
    weight[cell] += edge_length;
}

/// Turn an accumulated Σ(rate·mult·SCALE·edge_length) seed into an edge-length-
/// weighted MEAN rate × `FEATURE_FORCE_REF_SPACING`. The result is intensive —
/// independent of cell count — so feature amplitudes are resolution-invariant.
/// Cells with no contributing boundary edge stay at 0.
fn normalize_force_seed(strength: &mut [f32], weight: &[f32]) {
    for i in 0..strength.len() {
        strength[i] = if weight[i] > 0.0 {
            strength[i] / weight[i] * FEATURE_FORCE_REF_SPACING
        } else {
            0.0
        };
    }
}

fn distance_field_from_edge_seed_cells(
    tessellation: &Tessellation,
    plates: &Plates,
    seed_strength: &[f32],
    seed_dist0: &[f32],
    restrict_to_plate: bool,
) -> Vec<f32> {
    #[derive(Clone, Copy, PartialEq)]
    struct State {
        dist: f32,
        cell: usize,
        plate: u32,
    }

    impl Eq for State {}

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .dist
                .partial_cmp(&self.dist)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = tessellation.num_cells();
    let mut dist = vec![f32::INFINITY; n];
    let mut heap = BinaryHeap::new();

    for i in 0..n {
        let d0 = seed_dist0[i];
        if seed_strength[i] > 0.0 && d0.is_finite() {
            let plate = plates.cell_plate[i];
            dist[i] = dist[i].min(d0);
            heap.push(State {
                dist: d0,
                cell: i,
                plate,
            });
        }
    }

    const TIE_EPS: f32 = 1e-6;

    while let Some(State {
        dist: d,
        cell,
        plate,
    }) = heap.pop()
    {
        if d > dist[cell] + TIE_EPS {
            continue;
        }

        let pos = tessellation.cell_center(cell);

        for &neighbor in tessellation.neighbors(cell) {
            if restrict_to_plate && plates.cell_plate[neighbor] != plate {
                continue;
            }

            let neighbor_pos = tessellation.cell_center(neighbor);
            let step = angular_distance(pos, neighbor_pos);
            let nd = d + step;

            if nd + TIE_EPS < dist[neighbor] {
                dist[neighbor] = nd;
                heap.push(State {
                    dist: nd,
                    cell: neighbor,
                    plate,
                });
            }
        }
    }

    dist
}

/// Get convergent boundary multiplier for uplift-style features.
fn uplift_multiplier(my_type: CrustType, other_type: CrustType) -> f32 {
    match (my_type, other_type) {
        (CrustType::Continental, CrustType::Continental) => CONV_CONT_CONT,
        (CrustType::Oceanic, CrustType::Oceanic) => CONV_OCEAN_OCEAN,
        (CrustType::Continental, CrustType::Oceanic) => CONV_CONT_OCEAN,
        (CrustType::Oceanic, CrustType::Continental) => CONV_OCEAN_CONT,
    }
}

/// Get subduction multiplier for trench forcing (subducting side).
fn subduction_multiplier(subducting_type: CrustType, overriding_type: CrustType) -> f32 {
    match (subducting_type, overriding_type) {
        (CrustType::Oceanic, CrustType::Continental) => SUBD_OCEAN_CONT,
        (CrustType::Oceanic, CrustType::Oceanic) => SUBD_OCEAN_OCEAN,
        _ => 0.0,
    }
}

/// Compute distance field from seed cells using Dijkstra's algorithm.
///
/// Returns (distance in radians, source cell index) for each cell.
/// Seeds are cells where `seed_strength[i] > 0`.
#[allow(dead_code)]
pub fn distance_field_from_seeds(
    tessellation: &Tessellation,
    plates: &Plates,
    seed_strength: &[f32],
    restrict_to_plate: bool,
) -> (Vec<f32>, Vec<Option<usize>>) {
    #[derive(Clone, Copy, PartialEq)]
    struct State {
        dist: f32,
        cell: usize,
    }

    impl Eq for State {}

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .dist
                .partial_cmp(&self.dist)
                .unwrap_or(Ordering::Equal)
        }
    }

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let n = tessellation.num_cells();
    let mut dist = vec![f32::INFINITY; n];
    let mut src: Vec<Option<usize>> = vec![None; n];
    let mut heap = BinaryHeap::new();

    for i in 0..n {
        if seed_strength[i] > 0.0 {
            dist[i] = 0.0;
            src[i] = Some(i);
            heap.push(State { dist: 0.0, cell: i });
        }
    }

    while let Some(State { dist: d, cell }) = heap.pop() {
        if d > dist[cell] {
            continue;
        }

        let pos = tessellation.cell_center(cell);
        let plate = plates.cell_plate[cell];

        for &neighbor in tessellation.neighbors(cell) {
            if restrict_to_plate && plates.cell_plate[neighbor] != plate {
                continue;
            }

            let neighbor_pos = tessellation.cell_center(neighbor);
            let arc_dist = pos.dot(neighbor_pos).clamp(-1.0, 1.0).acos();
            let nd = d + arc_dist;

            if nd < dist[neighbor] {
                dist[neighbor] = nd;
                src[neighbor] = src[cell];
                heap.push(State {
                    dist: nd,
                    cell: neighbor,
                });
            }
        }
    }

    (dist, src)
}

fn solve_plate_screened_diffusion(
    tessellation: &Tessellation,
    plates: &Plates,
    rhs: &[f32],
    decay_length: f32,
    mean_neighbor_dist: f32,
) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    let num_plates = plates.num_plates;

    // Convert decay length to λ for screened diffusion
    let k = decay_length.max(0.0) / mean_neighbor_dist.max(1e-6);
    let lambda = k * k;

    // Adaptive max iterations: higher resolution needs more iterations to converge.
    // At ~10k cells, mean_neighbor_dist ≈ 0.06 rad. Scale proportionally.
    const REFERENCE_NEIGHBOR_DIST: f32 = 0.06;
    let resolution_scale = (REFERENCE_NEIGHBOR_DIST / mean_neighbor_dist).max(1.0);
    let adaptive_max_iters = ((DIFFUSION_MAX_ITERS as f32) * resolution_scale).ceil() as usize;

    // Build plate membership lists
    let mut plate_cells: Vec<Vec<usize>> = vec![Vec::new(); num_plates];
    for (cell_idx, &plate) in plates.cell_plate.iter().enumerate() {
        plate_cells[plate as usize].push(cell_idx);
    }

    let mut solution = vec![0.0f32; num_cells];

    // Solve independently per plate using Gauss-Seidel iteration
    for cells in &plate_cells {
        if cells.is_empty() {
            continue;
        }

        // Build local index mapping
        let mut global_to_local: Vec<usize> = vec![usize::MAX; num_cells];
        for (local_idx, &global_idx) in cells.iter().enumerate() {
            global_to_local[global_idx] = local_idx;
        }

        // Build plate-restricted adjacency and diagonal terms
        let mut local_neighbors: Vec<Vec<usize>> = Vec::with_capacity(cells.len());
        let mut diag: Vec<f32> = Vec::with_capacity(cells.len());

        for &global_idx in cells {
            let neighbors: Vec<usize> = tessellation
                .neighbors(global_idx)
                .iter()
                .filter(|&&n| plates.cell_plate[n] == plates.cell_plate[global_idx])
                .map(|&n| global_to_local[n])
                .collect();

            let degree = neighbors.len() as f32;
            diag.push(1.0 + lambda * degree);
            local_neighbors.push(neighbors);
        }

        // Initialize with RHS
        let mut s: Vec<f32> = cells.iter().map(|&i| rhs[i]).collect();
        let b: Vec<f32> = s.clone();

        // Gauss-Seidel iteration
        let omega = DIFFUSION_DAMPING;

        for _ in 0..adaptive_max_iters {
            let mut max_change: f32 = 0.0;

            for local_idx in 0..cells.len() {
                let neighbor_sum: f32 = local_neighbors[local_idx].iter().map(|&n| s[n]).sum();
                let candidate = (b[local_idx] + lambda * neighbor_sum) / diag[local_idx];
                let new_val = (1.0 - omega) * s[local_idx] + omega * candidate;

                max_change = max_change.max((new_val - s[local_idx]).abs());
                s[local_idx] = new_val;
            }

            if max_change < DIFFUSION_TOLERANCE {
                break;
            }
        }

        // Copy results back
        for (local_idx, &global_idx) in cells.iter().enumerate() {
            solution[global_idx] = s[local_idx];
        }
    }

    solution
}

/// Smooth boundary forcing without introducing an extra inland decay.
///
/// This performs a normalized diffusion: diffuse both the forcing and a unit "seed weight" field,
/// then divide. The ratio behaves like a local average of boundary forcing over the diffusion scale.
fn compute_smoothed_boundary_forcing(
    tessellation: &Tessellation,
    plates: &Plates,
    boundary_forcing: &[f32],
    support_dist: f32,
    mean_neighbor_dist: f32,
) -> Vec<f32> {
    let weight: Vec<f32> = boundary_forcing
        .iter()
        .map(|&s| if s > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let num = solve_plate_screened_diffusion(
        tessellation,
        plates,
        boundary_forcing,
        support_dist,
        mean_neighbor_dist,
    );
    let den = solve_plate_screened_diffusion(
        tessellation,
        plates,
        &weight,
        support_dist,
        mean_neighbor_dist,
    );

    num.iter()
        .zip(den.iter())
        .map(|(&n, &d)| if d > 1e-6 { (n / d).max(0.0) } else { 0.0 })
        .collect()
}

/// Compute a plate-constrained boundary influence scalar via screened diffusion.
///
/// Unlike the previous max-normalized approach, this uses a fixed physical scale derived from
/// `MAX_ANGULAR_VELOCITY` and the mean cell spacing so values are comparable across worlds.
fn compute_influence_field(
    tessellation: &Tessellation,
    plates: &Plates,
    boundary_forcing: &[f32],
    influence_length: f32,
    mean_neighbor_dist: f32,
) -> Vec<f32> {
    let raw = solve_plate_screened_diffusion(
        tessellation,
        plates,
        boundary_forcing,
        influence_length,
        mean_neighbor_dist,
    );

    // Reference magnitude:
    //
    // `boundary_forcing` is built from kinematic rates (e.g., convergence, shear) multiplied by a
    // boundary edge length, so a natural physical scale is (speed * typical edge length).
    //
    // However, the screened-diffusion solve attenuates localized sources roughly by a factor of
    // (1 + λ * degree), where λ = (influence_length / mean_neighbor_dist)^2 and `degree` is the
    // cell's neighbor count. Without accounting for this, normalized fields are systematically
    // too small (especially at larger influence lengths).
    let k = influence_length.max(0.0) / mean_neighbor_dist.max(1e-6);
    let lambda = k * k;

    let num_cells = tessellation.num_cells().max(1);
    let total_degree: usize = (0..num_cells)
        .map(|i| tessellation.neighbors(i).len())
        .sum();
    let mean_degree = (total_degree as f32 / num_cells as f32).max(1.0);

    let attenuation = 1.0 + lambda * mean_degree;
    let reference = (2.0 * MAX_ANGULAR_VELOCITY * mean_neighbor_dist / attenuation).max(1e-6);
    // Soft saturation (x / (x + ref)) instead of a hard clamp: forcing along
    // strong boundaries routinely exceeds the reference, and a hard clamp
    // flattens all along-strike variation (every strong rift segment reads
    // exactly 1.0, producing uniform features). Soft saturation keeps the
    // 0-1 range and the ordering at every magnitude.
    raw.iter().map(|&x| x / (x + reference)).collect()
}

fn oceanic_age_factor_from_ridge_distance(ridge_distance: f32) -> f32 {
    if !ridge_distance.is_finite() {
        return 1.0;
    }
    (ridge_distance / THERMAL_SUBSIDENCE_WIDTH)
        .sqrt()
        .clamp(0.0, 1.0)
}

fn ridge_age_distance_from_spreading_rate(ridge_distance: f32, spreading_rate: f32) -> f32 {
    if !ridge_distance.is_finite() {
        return f32::INFINITY;
    }
    if spreading_rate <= 1e-6 {
        return f32::INFINITY;
    }
    ridge_distance * (OCEAN_SPREADING_REFERENCE_RATE / spreading_rate)
}

/// Compute mean angular distance between neighboring cells.
fn compute_mean_neighbor_distance(tessellation: &Tessellation) -> f32 {
    let mut total_dist: f32 = 0.0;
    let mut count: usize = 0;

    for i in 0..tessellation.num_cells() {
        let pos_i = tessellation.cell_center(i);
        for &j in tessellation.neighbors(i) {
            if j > i {
                let pos_j = tessellation.cell_center(j);
                let dist = pos_i.dot(pos_j).clamp(-1.0, 1.0).acos();
                total_dist += dist;
                count += 1;
            }
        }
    }

    if count > 0 {
        total_dist / count as f32
    } else {
        0.03
    }
}

// --- Helper functions ---

/// Square root response function with sensitivity and maximum cap.
pub fn sqrt_response(value: f32, sensitivity: f32, max: f32) -> f32 {
    (value.max(0.0) * sensitivity).sqrt().min(max)
}

/// Gaussian band profile centered at `peak` with given `width`.
pub fn gaussian_band(dist: f32, peak: f32, width: f32) -> f32 {
    let w = width.max(1e-6);
    let z = (dist - peak) / w;
    (-0.5 * z * z).exp()
}

/// Exponential decay from distance.
pub fn exp_decay(dist: f32, decay: f32) -> f32 {
    let d = dist.max(0.0);
    let k = decay.max(1e-6);
    (-(d / k)).exp()
}

/// Broken elastic plate flexure profile for the subducting side.
///
/// Returns normalized downward deflection `w/w0` for an end-loaded broken
/// thin elastic plate: `exp(-d/alpha) * (cos(d/alpha) + sin(d/alpha))`.
pub fn flexure_broken(dist: f32, alpha: f32) -> f32 {
    if alpha <= 0.0 || !dist.is_finite() {
        return 0.0;
    }
    let x = dist.max(0.0) / alpha;
    (-x).exp() * (x.cos() + x.sin())
}

/// Coupled continuous-plate flexure profile for the overriding forearc.
///
/// Returns normalized downward deflection `w/w0` for the continuous-plate
/// member of the flexure family: `exp(-d/alpha) * cos(d/alpha)`.
pub fn flexure_coupled(dist: f32, alpha: f32) -> f32 {
    if alpha <= 0.0 || !dist.is_finite() {
        return 0.0;
    }
    let x = dist.max(0.0) / alpha;
    (-x).exp() * x.cos()
}

/// Smoothstep function for gradual transitions.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge0 == edge1 {
        // Degenerate case: step function at edge0
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn broken_flexure_matches_key_points() {
        let a = 0.018;

        assert!((flexure_broken(0.0, a) - 1.0).abs() < EPS);
        assert!(flexure_broken(2.35 * a, a) > 0.0);
        assert!(flexure_broken(2.37 * a, a) < 0.0);

        let outer_rise = flexure_broken(PI * a, a);
        assert!((outer_rise - -0.0432).abs() <= 0.05 * 0.0432);
        assert!(flexure_broken(2.01 * PI * a, a).abs() < 0.005);
    }

    #[test]
    fn coupled_flexure_matches_key_points() {
        let a = 0.015;

        assert!((flexure_coupled(0.0, a) - 1.0).abs() < EPS);
        assert!(flexure_coupled(0.49 * PI * a, a) > 0.0);
        assert!(flexure_coupled(0.51 * PI * a, a) < 0.0);

        let overshoot = flexure_coupled(0.75 * PI * a, a);
        assert!((overshoot - -0.0670).abs() <= 0.05 * 0.0670);
    }

    #[test]
    fn flexure_profiles_guard_invalid_inputs() {
        assert_eq!(flexure_broken(0.1, 0.0), 0.0);
        assert_eq!(flexure_broken(f32::INFINITY, 0.1), 0.0);
        assert_eq!(flexure_coupled(0.1, 0.0), 0.0);
        assert_eq!(flexure_coupled(f32::INFINITY, 0.1), 0.0);
    }

    #[test]
    fn spreading_rate_age_distance_recovers_reference_and_saturates_slow() {
        let d = 0.2;
        assert_eq!(
            ridge_age_distance_from_spreading_rate(d, OCEAN_SPREADING_REFERENCE_RATE),
            d
        );
        assert!(
            ridge_age_distance_from_spreading_rate(d, 2.0 * OCEAN_SPREADING_REFERENCE_RATE) < d
        );
        let stagnant = ridge_age_distance_from_spreading_rate(d, 0.0);
        assert!(!stagnant.is_finite());
        assert_eq!(oceanic_age_factor_from_ridge_distance(stagnant), 1.0);
    }

    /// Forcing must be INTENSIVE: a boundary cell's normalized seed depends only
    /// on the kinematic rate, not on its edge length or how many edges it has.
    /// This is what makes feature amplitudes resolution-invariant — refining the
    /// mesh shrinks each boundary cell's edge length, and the old area-scaled SUM
    /// shrank the per-cell forcing as ~edge_length (~1/sqrt(N)); the weighted MEAN
    /// does not.
    #[test]
    fn force_seed_normalization_is_intensive() {
        let rate = 0.5f32; // closing rate × mult × SCALE, the per-edge intensity
        let expected = rate * FEATURE_FORCE_REF_SPACING;

        // Coarse cell: one boundary edge of length L. force already folds in L.
        let l_coarse = 0.02f32;
        let (mut s_coarse, mut w_coarse) = (vec![0.0f32], vec![0.0f32]);
        add_force_seed(&mut s_coarse, &mut w_coarse, 0, rate * l_coarse, l_coarse);
        normalize_force_seed(&mut s_coarse, &w_coarse);

        // Refined cell: half the edge length (denser mesh), same rate.
        let l_fine = 0.01f32;
        let (mut s_fine, mut w_fine) = (vec![0.0f32], vec![0.0f32]);
        add_force_seed(&mut s_fine, &mut w_fine, 0, rate * l_fine, l_fine);
        normalize_force_seed(&mut s_fine, &w_fine);

        assert!(
            (s_coarse[0] - expected).abs() < 1e-6 && (s_fine[0] - expected).abs() < 1e-6,
            "normalized forcing must equal rate×REF regardless of edge length: \
             coarse={}, fine={}, expected={}",
            s_coarse[0],
            s_fine[0],
            expected
        );

        // Two equal-rate edges on one cell must AVERAGE (stay rate×REF), not SUM
        // (which the old formulation did, over-forcing multi-edge boundary cells).
        let (mut s_two, mut w_two) = (vec![0.0f32], vec![0.0f32]);
        add_force_seed(&mut s_two, &mut w_two, 0, rate * l_fine, l_fine);
        add_force_seed(&mut s_two, &mut w_two, 0, rate * l_fine, l_fine);
        normalize_force_seed(&mut s_two, &w_two);
        assert!(
            (s_two[0] - expected).abs() < 1e-6,
            "two equal-rate edges should average to rate×REF, got {}",
            s_two[0]
        );

        // A cell with no contributing edge stays at zero.
        let (mut s_zero, w_zero) = (vec![0.0f32], vec![0.0f32]);
        normalize_force_seed(&mut s_zero, &w_zero);
        assert_eq!(s_zero[0], 0.0);
    }
}
