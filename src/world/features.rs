//! Tectonic feature fields derived from plate boundaries.
//!
//! This module computes canonical per-cell fields (trench, arc, ridge, collision, activity, regime)
//! from plate boundary edges. These fields are resolution-independent (distances in radians)
//! and serve as the primary drivers of terrain elevation.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::BinaryHeap;

use glam::Vec3;

use super::boundary::{collect_plate_boundaries, BoundaryKind, SubductionPolarity};
use super::constants::*;
use super::dynamics::{Dynamics, PlateType};
use super::{Plates, Tessellation};

/// Tectonic feature fields derived from plate boundaries.
///
/// All values are resolution-independent magnitudes (not raw elevations).
/// Elevation generation applies these via decay functions and sensitivity constants.
pub struct FeatureFields {
    /// Trench depth field (oceanic subducting side only).
    /// Stores the computed depth magnitude (positive value = depression depth).
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
    /// Used for thermal subsidence calculation in elevation generation.
    /// Infinity for cells with no ridge on their plate.
    pub ridge_distance: Vec<f32>,
}

impl FeatureFields {
    /// Compute all feature fields from plate boundaries.
    pub fn compute(
        tessellation: &Tessellation,
        plates: &Plates,
        dynamics: &Dynamics,
    ) -> Self {
        let boundaries = collect_plate_boundaries(tessellation, plates, dynamics);
        let num_cells = tessellation.num_cells();
        let boundary_edge_midpoints = build_cell_pair_edge_midpoints(tessellation);

        // Build edge-anchored seed arrays for each feature type.
        //
        // We keep per-cell seed strengths for summing contributions, but the "source distance"
        // is the distance from the cell center to the shared Voronoi edge midpoint. This avoids
        // quantizing all features to "0 at boundary cell centers", which makes arcs/trenches
        // appear too close to boundaries at coarse resolutions.
        let mut trench_seed_strength = vec![0.0f32; num_cells];
        let mut trench_seed_dist0 = vec![f32::INFINITY; num_cells];

        let mut arc_seed_strength_cont = vec![0.0f32; num_cells];
        let mut arc_seed_dist0_cont = vec![f32::INFINITY; num_cells];

        let mut arc_seed_strength_ocean = vec![0.0f32; num_cells];
        let mut arc_seed_dist0_ocean = vec![f32::INFINITY; num_cells];

        let mut ridge_seed_strength_ocean = vec![0.0f32; num_cells];
        let mut ridge_seed_dist0_ocean = vec![f32::INFINITY; num_cells];

        let mut collision_seed_strength = vec![0.0f32; num_cells];
        let mut collision_seed_dist0 = vec![f32::INFINITY; num_cells];

        let mut activity_seed = vec![0.0f32; num_cells];
        let mut convergent_seed = vec![0.0f32; num_cells];
        let mut divergent_seed = vec![0.0f32; num_cells];
        let mut transform_seed = vec![0.0f32; num_cells];

        for b in &boundaries {
            // Activity: all boundary cells get activity based on relative speed
            let activity_force = b.relative_speed * b.edge_length;
            activity_seed[b.cell_a] += activity_force;
            activity_seed[b.cell_b] += activity_force;

            // Regime influence: split into convergent/divergent/transform boundary drivers.
            // These are magnitude-only kinematic weights used for noise modulation, not
            // for the feature magnitudes (those use `closing` etc below).
            match b.kind {
                BoundaryKind::Convergent => {
                    let closing = b.convergence.max(0.0);
                    let force = closing * b.edge_length;
                    convergent_seed[b.cell_a] += force;
                    convergent_seed[b.cell_b] += force;
                }
                BoundaryKind::Divergent => {
                    let opening = (-b.convergence).max(0.0);
                    let force = opening * b.edge_length;
                    divergent_seed[b.cell_a] += force;
                    divergent_seed[b.cell_b] += force;
                }
                BoundaryKind::Transform => {
                    let shear = b.shear.abs();
                    let force = shear * b.edge_length;
                    transform_seed[b.cell_a] += force;
                    transform_seed[b.cell_b] += force;
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
                    let mult_a = convergent_multiplier(b.type_a, b.type_b);
                    let mult_b = convergent_multiplier(b.type_b, b.type_a);
                    let force_a = closing * mult_a * b.edge_length * FEATURE_FORCE_SCALE;
                    let force_b = closing * mult_b * b.edge_length * FEATURE_FORCE_SCALE;

                    // Handle subduction (trench + arc) vs collision
                    if let Some(polarity) = b.subduction {
                        match polarity {
                            SubductionPolarity::ASubducts => {
                                // A subducts: trench on A if oceanic; arc on B (overriding)
                                if b.type_a == PlateType::Oceanic {
                                    trench_seed_strength[b.cell_a] += force_a;
                                    trench_seed_dist0[b.cell_a] = trench_seed_dist0[b.cell_a].min(dist0_a);
                                }
                                match b.type_b {
                                    PlateType::Continental => {
                                        arc_seed_strength_cont[b.cell_b] += force_b;
                                        arc_seed_dist0_cont[b.cell_b] =
                                            arc_seed_dist0_cont[b.cell_b].min(dist0_b);
                                    }
                                    PlateType::Oceanic => {
                                        arc_seed_strength_ocean[b.cell_b] += force_b;
                                        arc_seed_dist0_ocean[b.cell_b] =
                                            arc_seed_dist0_ocean[b.cell_b].min(dist0_b);
                                    }
                                }
                            }
                            SubductionPolarity::BSubducts => {
                                if b.type_b == PlateType::Oceanic {
                                    trench_seed_strength[b.cell_b] += force_b;
                                    trench_seed_dist0[b.cell_b] = trench_seed_dist0[b.cell_b].min(dist0_b);
                                }
                                match b.type_a {
                                    PlateType::Continental => {
                                        arc_seed_strength_cont[b.cell_a] += force_a;
                                        arc_seed_dist0_cont[b.cell_a] =
                                            arc_seed_dist0_cont[b.cell_a].min(dist0_a);
                                    }
                                    PlateType::Oceanic => {
                                        arc_seed_strength_ocean[b.cell_a] += force_a;
                                        arc_seed_dist0_ocean[b.cell_a] =
                                            arc_seed_dist0_ocean[b.cell_a].min(dist0_a);
                                    }
                                }
                            }
                        }
                    } else {
                        // No subduction polarity = continent-continent collision
                        if b.type_a == PlateType::Continental && b.type_b == PlateType::Continental {
                            collision_seed_strength[b.cell_a] += force_a;
                            collision_seed_dist0[b.cell_a] =
                                collision_seed_dist0[b.cell_a].min(dist0_a);
                            collision_seed_strength[b.cell_b] += force_b;
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
                    if b.type_a == PlateType::Oceanic && b.type_b == PlateType::Oceanic {
                        let force =
                            opening * DIV_OCEAN_OCEAN * b.edge_length * FEATURE_FORCE_SCALE;
                        ridge_seed_strength_ocean[b.cell_a] += force;
                        ridge_seed_dist0_ocean[b.cell_a] =
                            ridge_seed_dist0_ocean[b.cell_a].min(dist0_a);
                        ridge_seed_strength_ocean[b.cell_b] += force;
                        ridge_seed_dist0_ocean[b.cell_b] =
                            ridge_seed_dist0_ocean[b.cell_b].min(dist0_b);
                    }
                }
                BoundaryKind::Transform => {
                    // Transforms don't produce elevation features
                    // (activity already captured above)
                }
            }
        }

        // Compute edge-anchored distance + strength fields from seeds.
        let (trench_dist, trench_strength) = distance_strength_field_from_edge_seed_cells(
            tessellation,
            plates,
            &trench_seed_strength,
            &trench_seed_dist0,
            true,
        );
        let (arc_dist_cont, arc_strength_cont) = distance_strength_field_from_edge_seed_cells(
            tessellation,
            plates,
            &arc_seed_strength_cont,
            &arc_seed_dist0_cont,
            true,
        );
        let (arc_dist_ocean, arc_strength_ocean) = distance_strength_field_from_edge_seed_cells(
            tessellation,
            plates,
            &arc_seed_strength_ocean,
            &arc_seed_dist0_ocean,
            true,
        );
        let (ridge_dist, ridge_strength_ocean) = distance_strength_field_from_edge_seed_cells(
            tessellation,
            plates,
            &ridge_seed_strength_ocean,
            &ridge_seed_dist0_ocean,
            true,
        );
        let (collision_dist, collision_strength) = distance_strength_field_from_edge_seed_cells(
            tessellation,
            plates,
            &collision_seed_strength,
            &collision_seed_dist0,
            true,
        );

        // Compute activity and regime influence via diffusion (not distance-based, uses screened diffusion)
        let activity = compute_diffused_field(tessellation, plates, &activity_seed);
        let convergent = compute_diffused_field(tessellation, plates, &convergent_seed);
        let divergent = compute_diffused_field(tessellation, plates, &divergent_seed);
        let transform = compute_diffused_field(tessellation, plates, &transform_seed);

        // Lightly smooth strengths within the near-boundary band to avoid patchy amplitudes
        // from nearest-source partitioning.
        let mut trench_strength = trench_strength;
        let mut arc_strength_cont = arc_strength_cont;
        let mut arc_strength_ocean = arc_strength_ocean;
        let mut ridge_strength_ocean = ridge_strength_ocean;
        let mut collision_strength = collision_strength;

        const SMOOTH_ITERS: usize = 2;
        smooth_strength_in_band(
            tessellation,
            plates,
            &trench_dist,
            &mut trench_strength,
            4.0 * TRENCH_DECAY,
            SMOOTH_ITERS,
        );
        smooth_strength_in_band(
            tessellation,
            plates,
            &arc_dist_cont,
            &mut arc_strength_cont,
            ARC_CONT_PEAK_DIST + 3.0 * ARC_CONT_WIDTH,
            SMOOTH_ITERS,
        );
        smooth_strength_in_band(
            tessellation,
            plates,
            &arc_dist_ocean,
            &mut arc_strength_ocean,
            ARC_OCEAN_PEAK_DIST + 3.0 * ARC_OCEAN_WIDTH,
            SMOOTH_ITERS,
        );
        smooth_strength_in_band(
            tessellation,
            plates,
            &ridge_dist,
            &mut ridge_strength_ocean,
            4.0 * RIDGE_DECAY,
            SMOOTH_ITERS,
        );
        smooth_strength_in_band(
            tessellation,
            plates,
            &collision_dist,
            &mut collision_strength,
            COLLISION_PEAK_DIST + 3.0 * COLLISION_WIDTH,
            SMOOTH_ITERS,
        );

        // Convert distance fields to feature magnitudes
        let mut trench = vec![0.0f32; num_cells];
        let mut arc = vec![0.0f32; num_cells];
        let mut ridge = vec![0.0f32; num_cells];
        let mut collision = vec![0.0f32; num_cells];

        for i in 0..num_cells {
            let plate_type = dynamics.plate_type(plates.cell_plate[i] as usize);
            let is_continental = plate_type == PlateType::Continental;

            // Trench: oceanic only
            if plate_type == PlateType::Oceanic {
                let d = trench_dist[i];
                if d.is_finite() {
                    let depth =
                        sqrt_response(trench_strength[i], TRENCH_SENSITIVITY, TRENCH_MAX_DEPTH);
                    trench[i] = depth * exp_decay(d, TRENCH_DECAY);
                }
            }

            // Arc uplift: continental or oceanic depending on plate type
            let (arc_dist_val, strength) = if is_continental {
                (arc_dist_cont[i], arc_strength_cont[i])
            } else {
                (arc_dist_ocean[i], arc_strength_ocean[i])
            };
            if arc_dist_val.is_finite() {
                let (sens, max_uplift, peak, width, gap) = if is_continental {
                    (
                        ARC_CONT_SENSITIVITY,
                        ARC_CONT_MAX_UPLIFT,
                        ARC_CONT_PEAK_DIST,
                        ARC_CONT_WIDTH,
                        ARC_CONT_GAP,
                    )
                } else {
                    (
                        ARC_OCEAN_SENSITIVITY,
                        ARC_OCEAN_MAX_UPLIFT,
                        ARC_OCEAN_PEAK_DIST,
                        ARC_OCEAN_WIDTH,
                        ARC_OCEAN_GAP,
                    )
                };
                let uplift = sqrt_response(strength, sens, max_uplift);
                arc[i] = uplift * gaussian_band(arc_dist_val, peak, width, gap);
            }

            // Ridge: oceanic only
            if plate_type == PlateType::Oceanic {
                let d = ridge_dist[i];
                if d.is_finite() {
                    let uplift = sqrt_response(
                        ridge_strength_ocean[i],
                        RIDGE_SENSITIVITY,
                        RIDGE_MAX_UPLIFT,
                    );
                    ridge[i] = uplift * exp_decay(d, RIDGE_DECAY);
                }
            }

            // Collision: continental only
            if is_continental {
                let d = collision_dist[i];
                if d.is_finite() {
                    let uplift = sqrt_response(
                        collision_strength[i],
                        COLLISION_SENSITIVITY,
                        COLLISION_MAX_UPLIFT,
                    );
                    collision[i] =
                        uplift * gaussian_band(d, COLLISION_PEAK_DIST, COLLISION_WIDTH, 0.0);
                }
            }
        }

        Self {
            trench,
            arc,
            ridge,
            collision,
            activity,
            convergent,
            divergent,
            transform,
            ridge_distance: ridge_dist,
        }
    }
}

fn smooth_strength_in_band(
    tessellation: &Tessellation,
    plates: &Plates,
    dist: &[f32],
    strength: &mut [f32],
    max_dist: f32,
    iters: usize,
) {
    if iters == 0 || max_dist <= 0.0 {
        return;
    }

    let n = tessellation.num_cells();
    let max_dist = max_dist.max(0.0);

    for _ in 0..iters {
        let prev = strength.to_vec();
        for i in 0..n {
            if !dist[i].is_finite() || dist[i] > max_dist {
                continue;
            }
            let plate = plates.cell_plate[i];
            let mut sum = prev[i];
            let mut count = 1.0f32;
            for &nb in tessellation.neighbors(i) {
                if plates.cell_plate[nb] != plate {
                    continue;
                }
                if !dist[nb].is_finite() || dist[nb] > max_dist {
                    continue;
                }
                sum += prev[nb];
                count += 1.0;
            }
            strength[i] = sum / count;
        }
    }
}

fn angular_distance(a: Vec3, b: Vec3) -> f32 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

fn build_cell_pair_edge_midpoints(tessellation: &Tessellation) -> HashMap<(usize, usize), Vec3> {
    // Map Voronoi edges (vertex pairs) -> cells containing that edge, then produce a
    // cell-pair map keyed by (min_cell, max_cell) to the edge midpoint.
    let voronoi = &tessellation.voronoi;
    let mut edge_to_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for (cell_idx, cell) in voronoi.cells.iter().enumerate() {
        let verts = &cell.vertex_indices;
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

        let v0 = voronoi.vertices[va];
        let v1 = voronoi.vertices[vb];
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

fn distance_strength_field_from_edge_seed_cells(
    tessellation: &Tessellation,
    plates: &Plates,
    seed_strength: &[f32],
    seed_dist0: &[f32],
    restrict_to_plate: bool,
) -> (Vec<f32>, Vec<f32>) {
    #[derive(Clone, Copy, PartialEq)]
    struct State {
        dist: f32,
        cell: usize,
        strength: f32,
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
    let mut strength = vec![0.0f32; n];
    let mut heap = BinaryHeap::new();

    for i in 0..n {
        let s = seed_strength[i];
        let d0 = seed_dist0[i];
        if s > 0.0 && d0.is_finite() {
            let plate = plates.cell_plate[i];
            dist[i] = dist[i].min(d0);
            strength[i] = strength[i].max(s);
            heap.push(State {
                dist: d0,
                cell: i,
                strength: s,
                plate,
            });
        }
    }

    const TIE_EPS: f32 = 1e-6;

    while let Some(State {
        dist: d,
        cell,
        strength: s,
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

            if nd + TIE_EPS < dist[neighbor] || ((nd - dist[neighbor]).abs() <= TIE_EPS && s > strength[neighbor]) {
                dist[neighbor] = nd;
                strength[neighbor] = s;
                heap.push(State {
                    dist: nd,
                    cell: neighbor,
                    strength: s,
                    plate,
                });
            }
        }
    }

    (dist, strength)
}

/// Get convergent boundary multiplier for a given plate type configuration.
fn convergent_multiplier(my_type: PlateType, other_type: PlateType) -> f32 {
    match (my_type, other_type) {
        (PlateType::Continental, PlateType::Continental) => CONV_CONT_CONT,
        (PlateType::Oceanic, PlateType::Oceanic) => CONV_OCEAN_OCEAN,
        (PlateType::Continental, PlateType::Oceanic) => CONV_CONT_OCEAN,
        (PlateType::Oceanic, PlateType::Continental) => CONV_OCEAN_CONT,
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
                heap.push(State { dist: nd, cell: neighbor });
            }
        }
    }

    (dist, src)
}

/// Compute a plate-constrained boundary influence scalar via screened diffusion.
///
/// The input is a per-cell seed strength (typically sourced from boundary edges).
/// The output is normalized to 0-1 range and diffused only within each plate.
fn compute_diffused_field(
    tessellation: &Tessellation,
    plates: &Plates,
    boundary_activity: &[f32],
) -> Vec<f32> {
    let num_cells = tessellation.num_cells();
    let num_plates = plates.num_plates;

    // Compute mean neighbor distance to calibrate decay
    let mean_neighbor_dist = compute_mean_neighbor_distance(tessellation);

    // Convert decay length to λ for screened diffusion
    let k = ACTIVITY_DECAY_LENGTH / mean_neighbor_dist;
    let lambda = k * k;

    // Build plate membership lists
    let mut plate_cells: Vec<Vec<usize>> = vec![Vec::new(); num_plates];
    for (cell_idx, &plate) in plates.cell_plate.iter().enumerate() {
        plate_cells[plate as usize].push(cell_idx);
    }

    let mut activity = vec![0.0f32; num_cells];

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

        // Initialize with boundary activity
        let mut s: Vec<f32> = cells.iter().map(|&i| boundary_activity[i]).collect();
        let b: Vec<f32> = cells.iter().map(|&i| boundary_activity[i]).collect();

        // Gauss-Seidel iteration
        let omega = DIFFUSION_DAMPING;

        for _ in 0..DIFFUSION_MAX_ITERS {
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
            activity[global_idx] = s[local_idx];
        }
    }

    // Normalize to 0-1 range
    let max_val = activity.iter().cloned().fold(0.0f32, f32::max);
    if max_val > 0.0 {
        for a in &mut activity {
            *a = (*a / max_val).min(1.0);
        }
    }

    activity
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

/// Gaussian band profile with peak offset, width, and near-boundary gap.
pub fn gaussian_band(dist: f32, peak: f32, width: f32, gap: f32) -> f32 {
    // Suppress directly at the boundary (forearc gap) so features don't sit right on the interface
    let gap_t = smoothstep(0.0, gap, dist);
    let w = width.max(1e-6);
    let z = (dist - peak) / w;
    gap_t * (-0.5 * z * z).exp()
}

/// Exponential decay from distance.
pub fn exp_decay(dist: f32, decay: f32) -> f32 {
    let d = dist.max(0.0);
    let k = decay.max(1e-6);
    (-(d / k)).exp()
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
