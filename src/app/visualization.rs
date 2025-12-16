//! Visualization generators for tectonic plate overlays.
//!
//! This module generates visual elements (arrows, markers, colored edges)
//! for rendering plate tectonics data, separated from the domain models.

use std::collections::{HashMap, HashSet};

use glam::Vec3;
use rand::Rng;

use hex3::world::{Tessellation, World};

use super::coloring::hsl_to_rgb;

/// Build a map of boundary edge colors keyed by (vertex_a, vertex_b) in canonical order.
/// Non-boundary edges won't be in the map (use default dark color).
pub fn build_boundary_edge_colors(world: &World) -> HashMap<(usize, usize), Vec3> {
    let plates = world.plates.as_ref().expect("Plates must be generated");
    let dynamics = world.dynamics.as_ref().expect("Dynamics must be generated");
    let tessellation = &world.tessellation;
    let voronoi = &tessellation.voronoi;

    let mut edge_colors: HashMap<(usize, usize), Vec3> = HashMap::new();
    let mut processed_cell_pairs: HashSet<(usize, usize)> = HashSet::new();

    for (cell_idx, neighbors) in tessellation.adjacency.iter().enumerate() {
        let plate_a = plates.cell_plate[cell_idx] as usize;

        for &neighbor_idx in neighbors {
            let plate_b = plates.cell_plate[neighbor_idx] as usize;

            if plate_a == plate_b {
                continue;
            }

            let cell_pair = if cell_idx < neighbor_idx {
                (cell_idx, neighbor_idx)
            } else {
                (neighbor_idx, cell_idx)
            };

            if processed_cell_pairs.contains(&cell_pair) {
                continue;
            }
            processed_cell_pairs.insert(cell_pair);

            let cell_verts: HashSet<usize> = voronoi.cells[cell_idx]
                .vertex_indices
                .iter()
                .copied()
                .collect();
            let neighbor_verts: HashSet<usize> = voronoi.cells[neighbor_idx]
                .vertex_indices
                .iter()
                .copied()
                .collect();

            let shared: Vec<usize> = cell_verts.intersection(&neighbor_verts).copied().collect();

            if shared.len() != 2 {
                continue;
            }

            let edge_key = if shared[0] < shared[1] {
                (shared[0], shared[1])
            } else {
                (shared[1], shared[0])
            };

            let cell_pos = tessellation.cell_center(cell_idx);
            let neighbor_pos = tessellation.cell_center(neighbor_idx);
            let boundary_point = (cell_pos + neighbor_pos).normalize();

            let vel_a = dynamics.euler_pole(plate_a).velocity_at(boundary_point);
            let vel_b = dynamics.euler_pole(plate_b).velocity_at(boundary_point);
            let relative_vel = vel_a - vel_b;

            let chord = neighbor_pos - cell_pos;
            let tangent_normal = chord - boundary_point * chord.dot(boundary_point);
            let tangent_normal = if tangent_normal.length_squared() > 1e-10 {
                tangent_normal.normalize()
            } else {
                let up = if boundary_point.y.abs() < 0.9 {
                    Vec3::Y
                } else {
                    Vec3::X
                };
                boundary_point.cross(up).normalize()
            };

            let convergence = relative_vel.dot(tangent_normal);
            let stress = convergence;

            let color = if stress > 0.05 {
                let t = (stress / 0.5).clamp(0.0, 1.0);
                Vec3::new(0.7, 0.15, 0.1).lerp(Vec3::new(1.0, 0.2, 0.1), t)
            } else if stress < -0.05 {
                let t = (-stress / 0.3).clamp(0.0, 1.0);
                Vec3::new(0.1, 0.2, 0.7).lerp(Vec3::new(0.2, 0.4, 1.0), t)
            } else {
                Vec3::new(0.7, 0.7, 0.2)
            };

            edge_colors.insert(edge_key, color);
        }
    }

    edge_colors
}

/// Generate velocity arrow line segments for boundary cells.
/// Returns line segments: shaft + two arrowhead lines per arrow.
/// Each element is (start, end, color).
pub fn generate_velocity_arrows(world: &World) -> Vec<(Vec3, Vec3, Vec3)> {
    let plates = world.plates.as_ref().expect("Plates must be generated");
    let dynamics = world.dynamics.as_ref().expect("Dynamics must be generated");
    let tessellation = &world.tessellation;

    let arrow_length = 0.025;
    let head_length = 0.008;
    let head_angle: f32 = 0.4;
    let mut arrows = Vec::new();

    for &cell_idx in &plates.boundary_cells {
        let center = tessellation.cell_center(cell_idx);
        let plate_id = plates.cell_plate[cell_idx] as usize;
        let velocity = dynamics.euler_pole(plate_id).velocity_at(center);

        let speed = velocity.length();
        if speed < 0.001 {
            continue;
        }

        let dir = velocity.normalize();
        let tip = (center + dir * arrow_length).normalize();

        let max_speed = 1.0;
        let t = (speed / max_speed).clamp(0.0, 1.0);
        let hue = 240.0 * (1.0 - t);
        let color = hsl_to_rgb(hue, 0.9, 0.6);

        arrows.push((center, tip, color));

        let perp = center.cross(dir).normalize();
        let back_dir = -dir;
        let barb1_dir = (back_dir * head_angle.cos() + perp * head_angle.sin()).normalize();
        let barb2_dir = (back_dir * head_angle.cos() - perp * head_angle.sin()).normalize();

        let barb1_end = (tip + barb1_dir * head_length).normalize();
        let barb2_end = (tip + barb2_dir * head_length).normalize();

        arrows.push((tip, barb1_end, color));
        arrows.push((tip, barb2_end, color));
    }

    arrows
}

/// Generate pole marker triangles for each Euler pole.
/// Returns triangle vertices (3 points per pole) with (position, normal, color).
pub fn generate_pole_markers(world: &World) -> Vec<(Vec3, Vec3, Vec3)> {
    let plates = world.plates.as_ref().expect("Plates must be generated");
    let dynamics = world.dynamics.as_ref().expect("Dynamics must be generated");

    let marker_size = 0.04;
    let lift = 1.01;
    let mut markers = Vec::new();

    for (plate_id, pole) in dynamics.euler_poles.iter().enumerate() {
        let center = pole.axis.normalize();

        let up = if center.y.abs() < 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let tangent1 = center.cross(up).normalize();
        let tangent2 = center.cross(tangent1).normalize();

        let angle1 = 0.0_f32;
        let angle2 = std::f32::consts::TAU / 3.0;
        let angle3 = 2.0 * std::f32::consts::TAU / 3.0;

        let p1 = (center + (tangent1 * angle1.cos() + tangent2 * angle1.sin()) * marker_size)
            .normalize()
            * lift;
        let p2 = (center + (tangent1 * angle2.cos() + tangent2 * angle2.sin()) * marker_size)
            .normalize()
            * lift;
        let p3 = (center + (tangent1 * angle3.cos() + tangent2 * angle3.sin()) * marker_size)
            .normalize()
            * lift;

        let normal = center;
        let hue = (plate_id as f32 / plates.num_plates as f32) * 360.0;
        let color = hsl_to_rgb(hue, 1.0, 0.8);

        markers.push((p1, normal, color));
        markers.push((p2, normal, color));
        markers.push((p3, normal, color));
    }

    markers
}

/// Wind particle for flow visualization.
#[derive(Clone, Copy)]
struct WindParticle {
    /// Current position on unit sphere.
    pos: Vec3,
    /// Previous position (for drawing trail).
    prev_pos: Vec3,
    /// Current cell index (for wind lookup).
    cell_idx: usize,
    /// Age in frames (respawn when too old).
    age: u32,
}

/// Wind particle system for visualizing atmospheric flow.
pub struct WindParticles {
    particles: Vec<WindParticle>,
    /// Maximum particle age before respawn.
    max_age: u32,
    /// Time step for advection.
    dt: f32,
}

impl WindParticles {
    /// Number of particles to simulate.
    const NUM_PARTICLES: usize = 10000;

    /// Create a new wind particle system.
    pub fn new<R: Rng>(tessellation: &Tessellation, rng: &mut R) -> Self {
        let particles = (0..Self::NUM_PARTICLES)
            .map(|_| Self::spawn_particle(tessellation, rng))
            .collect();

        Self {
            particles,
            max_age: 120, // ~10 seconds at 60fps
            dt: 0.01,    // very slow movement for gentle flow visualization
        }
    }

    /// Spawn a particle at a random cell.
    fn spawn_particle<R: Rng>(tessellation: &Tessellation, rng: &mut R) -> WindParticle {
        let cell_idx = rng.gen_range(0..tessellation.num_cells());
        let pos = tessellation.cell_center(cell_idx);
        WindParticle {
            pos,
            prev_pos: pos,
            cell_idx,
            age: rng.gen_range(0..60), // stagger initial ages
        }
    }

    /// Update all particles by advecting with wind field.
    pub fn update<R: Rng>(&mut self, tessellation: &Tessellation, wind: &[Vec3], rng: &mut R) {
        for particle in &mut self.particles {
            // Store previous position for trail
            particle.prev_pos = particle.pos;

            // Get wind at current cell
            let cell_wind = wind[particle.cell_idx];

            // Advect: move along wind direction, project back to sphere
            let new_pos = (particle.pos + cell_wind * self.dt).normalize();
            particle.pos = new_pos;

            // Update cell tracking - check if we're closer to a neighbor
            particle.cell_idx = find_nearest_cell_local(tessellation, particle.pos, particle.cell_idx);

            // Age and respawn
            particle.age += 1;
            if particle.age >= self.max_age {
                *particle = Self::spawn_particle(tessellation, rng);
            }
        }
    }

    /// Generate line vertices for rendering particle trails.
    /// Returns (start, end, color) for each trail segment.
    pub fn generate_trail_lines(&self) -> Vec<(Vec3, Vec3, Vec3)> {
        self.particles
            .iter()
            .filter(|p| p.age > 0) // Skip just-spawned particles
            .filter(|p| (p.pos - p.prev_pos).length_squared() > 1e-8) // Skip stationary
            .map(|p| {
                // Fade based on age (young = bright, old = dim)
                let age_factor = 1.0 - (p.age as f32 / self.max_age as f32);
                let alpha = age_factor.powf(0.5); // sqrt for slower fade

                // Cyan/white color for wind
                let color = Vec3::new(0.3 + 0.7 * alpha, 0.8 + 0.2 * alpha, 1.0) * alpha;

                // Lift slightly above surface to prevent z-fighting
                let lift = 1.002;
                (p.prev_pos * lift, p.pos * lift, color)
            })
            .collect()
    }
}

/// Find the nearest cell using local search from a starting cell.
/// More efficient than brute force when particles move smoothly.
fn find_nearest_cell_local(tessellation: &Tessellation, pos: Vec3, start_cell: usize) -> usize {
    let mut best_cell = start_cell;
    let mut best_dist = (pos - tessellation.cell_center(start_cell)).length_squared();

    // Check neighbors of current cell
    for &neighbor in tessellation.neighbors(start_cell) {
        let dist = (pos - tessellation.cell_center(neighbor)).length_squared();
        if dist < best_dist {
            best_dist = dist;
            best_cell = neighbor;
        }
    }

    best_cell
}
