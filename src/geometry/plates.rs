use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use glam::Vec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use super::tectonics::{
    assign_plate_types_by_coverage_with_rng, calculate_boundary_stress, constants,
    elevation_to_color, generate_euler_poles_with_rng, generate_heightmap_with_noise,
    propagate_stress, EulerPole, PlateType,
};
use super::SphericalVoronoi;

/// Result of partitioning Voronoi cells into tectonic plates.
pub struct TectonicPlates {
    /// For each cell index, which plate does it belong to.
    pub cell_plate: Vec<u32>,
    /// Number of plates.
    pub num_plates: usize,
    /// Type of each plate (continental or oceanic).
    pub plate_types: Vec<PlateType>,
    /// Euler pole for each plate (rotation axis + angular velocity).
    pub euler_poles: Vec<EulerPole>,
    /// Boundary stress at each cell (before propagation).
    pub boundary_stress: Vec<f32>,
    /// Propagated stress at each cell (positive = compression, negative = tension).
    pub cell_stress: Vec<f32>,
    /// Elevation at each cell.
    pub cell_elevation: Vec<f32>,
    /// Noise contribution at each cell (for visualization).
    pub cell_noise: Vec<f32>,
    /// Indices of cells that are on plate boundaries.
    pub boundary_cells: Vec<usize>,
    /// Cell adjacency graph (neighbors for each cell).
    pub adjacency: Vec<Vec<usize>>,
}

impl TectonicPlates {
    /// Generate tectonic plates from a Voronoi diagram using flood fill.
    ///
    /// 1. Build cell adjacency graph from shared edges
    /// 2. Select random seed cells (one per plate)
    /// 3. Flood fill outward from all seeds simultaneously (BFS)
    /// 4. Assign plate types (oceanic vs continental)
    /// 5. Generate Euler poles for plate motion
    /// 6. Calculate boundary stress from relative plate velocities
    /// 7. Propagate stress inward with exponential decay
    /// 8. Generate heightmap from stress and plate types
    pub fn generate(voronoi: &SphericalVoronoi, num_plates: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::generate_with_rng(voronoi, num_plates, &mut rng)
    }

    /// Generate tectonic plates using a provided RNG for reproducibility.
    pub fn generate_with_rng<R: Rng>(
        voronoi: &SphericalVoronoi,
        num_plates: usize,
        rng: &mut R,
    ) -> Self {
        // Build adjacency graph
        let adjacency = build_adjacency(voronoi);

        // Select spaced seed cells (minimum angular distance between seeds)
        let seeds = select_seeds_spaced(voronoi, num_plates, rng);

        // Generate target sizes for varied plate sizes
        let target_sizes = generate_target_sizes(num_plates, rng);

        // Weighted flood fill: target_size controls size, noise creates irregular boundaries
        let cell_plate = flood_fill_weighted(&adjacency, &seeds, &target_sizes, voronoi, rng);

        // Assign plate types (larger plates more likely continental)
        let plate_types =
            assign_plate_types_by_coverage_with_rng(&cell_plate, num_plates, &target_sizes, rng);

        // Generate random Euler poles for plate motion
        let euler_poles = generate_euler_poles_with_rng(num_plates, rng);

        // Calculate stress at plate boundaries
        let boundary_stress =
            calculate_boundary_stress(&adjacency, &cell_plate, &euler_poles, &plate_types, voronoi);

        // Propagate stress inward with decay (plate-constrained sum model)
        let cell_stress = propagate_stress(&boundary_stress, &cell_plate, voronoi);

        // Create fBm for elevation noise (seeded from rng for reproducibility)
        let elevation_fbm: Fbm<Perlin> =
            Fbm::new(rng.gen()).set_octaves(constants::ELEVATION_NOISE_OCTAVES);

        // Generate heightmap from stress and plate types with noise
        let (cell_elevation, cell_noise) = generate_heightmap_with_noise(
            &cell_stress,
            &cell_plate,
            &plate_types,
            voronoi,
            &elevation_fbm,
        );

        // Find boundary cells (cells adjacent to cells on different plates)
        let boundary_cells = find_boundary_cells(&adjacency, &cell_plate);

        TectonicPlates {
            cell_plate,
            num_plates,
            plate_types,
            euler_poles,
            boundary_stress,
            cell_stress,
            cell_elevation,
            cell_noise,
            boundary_cells,
            adjacency,
        }
    }

    /// Get the color for a given cell based on its elevation (hypsometric tinting).
    pub fn cell_color_elevation(&self, cell_idx: usize) -> Vec3 {
        elevation_to_color(self.cell_elevation[cell_idx])
    }

    /// Get the color for a given cell based on its noise contribution.
    /// Green = positive noise (uplift), magenta = negative noise (depression).
    pub fn cell_color_noise(&self, cell_idx: usize) -> Vec3 {
        let noise = self.cell_noise[cell_idx];
        // Scale to visible range (noise is typically small, ~0.01 to 0.1)
        let t = (noise.abs() * 10.0).min(1.0);
        if noise >= 0.0 {
            // Positive: gray to green
            Vec3::new(0.3 * (1.0 - t), 0.3 + 0.7 * t, 0.3 * (1.0 - t))
        } else {
            // Negative: gray to magenta
            Vec3::new(0.3 + 0.7 * t, 0.3 * (1.0 - t), 0.3 + 0.7 * t)
        }
    }

    /// Get the color for a given cell based on its plate assignment.
    /// Continental plates use warm earth tones, oceanic plates use cool blue tones.
    pub fn cell_color_plate(&self, cell_idx: usize) -> Vec3 {
        let plate_id = self.cell_plate[cell_idx] as usize;
        let plate_type = self.plate_types[plate_id];

        // Count plates of each type to distribute hues within each palette
        let continental_plates: Vec<usize> = (0..self.num_plates)
            .filter(|&p| self.plate_types[p] == PlateType::Continental)
            .collect();
        let oceanic_plates: Vec<usize> = (0..self.num_plates)
            .filter(|&p| self.plate_types[p] == PlateType::Oceanic)
            .collect();

        match plate_type {
            PlateType::Continental => {
                // Warm palette: yellows, oranges, browns, olive greens (hue 30-90)
                let idx = continental_plates
                    .iter()
                    .position(|&p| p == plate_id)
                    .unwrap_or(0);
                let t = if continental_plates.len() > 1 {
                    idx as f32 / (continental_plates.len() - 1) as f32
                } else {
                    0.5
                };
                let hue = 30.0 + t * 60.0; // 30° (orange) to 90° (yellow-green)
                hsl_to_rgb(hue, 0.5, 0.5)
            }
            PlateType::Oceanic => {
                // Cool palette: teals, blues, cyans (hue 180-240)
                let idx = oceanic_plates
                    .iter()
                    .position(|&p| p == plate_id)
                    .unwrap_or(0);
                let t = if oceanic_plates.len() > 1 {
                    idx as f32 / (oceanic_plates.len() - 1) as f32
                } else {
                    0.5
                };
                let hue = 180.0 + t * 60.0; // 180° (cyan) to 240° (blue)
                hsl_to_rgb(hue, 0.5, 0.4)
            }
        }
    }

    /// Get the color for a given cell based on its stress value.
    /// Red = compression (positive), Blue = tension (negative), Gray = neutral.
    pub fn cell_color_stress(&self, cell_idx: usize) -> Vec3 {
        let stress = self.cell_stress[cell_idx];

        // Single-field visualization:
        // Red = compression (positive stress)
        // Blue = tension (negative stress)
        if stress > 0.0 {
            let t = (stress / 0.5).clamp(0.0, 1.0);
            Vec3::new(0.2 + t * 0.8, 0.2, 0.2)
        } else {
            let t = (-stress / 0.3).clamp(0.0, 1.0);
            Vec3::new(0.2, 0.2, 0.2 + t * 0.8)
        }
    }

    /// Legacy method - defaults to elevation coloring.
    pub fn cell_color(&self, cell_idx: usize) -> Vec3 {
        self.cell_color_elevation(cell_idx)
    }

    /// Build a map of boundary edge colors keyed by (vertex_a, vertex_b) in canonical order.
    /// Non-boundary edges won't be in the map (use default dark color).
    pub fn build_boundary_edge_colors(
        &self,
        voronoi: &SphericalVoronoi,
    ) -> HashMap<(usize, usize), Vec3> {
        let mut edge_colors: HashMap<(usize, usize), Vec3> = HashMap::new();
        let mut processed_cell_pairs: HashSet<(usize, usize)> = HashSet::new();

        // For each pair of adjacent cells on different plates
        for (cell_idx, neighbors) in self.adjacency.iter().enumerate() {
            let plate_a = self.cell_plate[cell_idx] as usize;

            for &neighbor_idx in neighbors {
                let plate_b = self.cell_plate[neighbor_idx] as usize;

                if plate_a == plate_b {
                    continue;
                }

                // Canonical cell pair to avoid processing twice
                let cell_pair = if cell_idx < neighbor_idx {
                    (cell_idx, neighbor_idx)
                } else {
                    (neighbor_idx, cell_idx)
                };

                if processed_cell_pairs.contains(&cell_pair) {
                    continue;
                }
                processed_cell_pairs.insert(cell_pair);

                // Find shared vertices (the edge between these cells)
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

                let shared: Vec<usize> =
                    cell_verts.intersection(&neighbor_verts).copied().collect();

                if shared.len() != 2 {
                    continue;
                }

                // Canonical edge key
                let edge_key = if shared[0] < shared[1] {
                    (shared[0], shared[1])
                } else {
                    (shared[1], shared[0])
                };

                // Calculate convergence for this edge
                let cell_pos = voronoi.generators[cell_idx];
                let neighbor_pos = voronoi.generators[neighbor_idx];
                let boundary_point = (cell_pos + neighbor_pos).normalize();

                let vel_a = self.euler_poles[plate_a].velocity_at(boundary_point);
                let vel_b = self.euler_poles[plate_b].velocity_at(boundary_point);
                let relative_vel = vel_a - vel_b;

                // Boundary direction: project chord into tangent plane at boundary_point
                let chord = neighbor_pos - cell_pos;
                let tangent_normal = chord - boundary_point * chord.dot(boundary_point);
                let tangent_normal = if tangent_normal.length_squared() > 1e-10 {
                    tangent_normal.normalize()
                } else {
                    // Degenerate case: use arbitrary tangent
                    let up = if boundary_point.y.abs() < 0.9 {
                        Vec3::Y
                    } else {
                        Vec3::X
                    };
                    boundary_point.cross(up).normalize()
                };
                // If A moves toward B (relative_vel aligns with tangent_normal) → positive (convergent)
                // If A moves away from B → negative (divergent)
                let convergence = relative_vel.dot(tangent_normal);

                // Edge color shows motion type (convergent/divergent), not plate-type-specific stress
                let stress = convergence;

                // Color by stress
                let color = if stress > 0.05 {
                    // Convergent - red
                    let t = (stress / 0.5).clamp(0.0, 1.0);
                    Vec3::new(0.7, 0.15, 0.1).lerp(Vec3::new(1.0, 0.2, 0.1), t)
                } else if stress < -0.05 {
                    // Divergent - blue
                    let t = (-stress / 0.3).clamp(0.0, 1.0);
                    Vec3::new(0.1, 0.2, 0.7).lerp(Vec3::new(0.2, 0.4, 1.0), t)
                } else {
                    // Neutral/transform - yellow
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
    pub fn generate_velocity_arrows(&self, voronoi: &SphericalVoronoi) -> Vec<(Vec3, Vec3, Vec3)> {
        let arrow_length = 0.025; // Shorter arrow shaft
        let head_length = 0.008; // Arrowhead length
        let head_angle: f32 = 0.4; // Arrowhead spread angle (radians)
        let mut arrows = Vec::new();

        for &cell_idx in &self.boundary_cells {
            let center = voronoi.generators[cell_idx];
            let plate_id = self.cell_plate[cell_idx] as usize;
            let velocity = self.euler_poles[plate_id].velocity_at(center);
            let _ = plate_id; // Used only for velocity lookup

            let speed = velocity.length();
            if speed < 0.001 {
                continue; // Skip near-stationary cells
            }

            // Direction of arrow (tangent to sphere)
            let dir = velocity.normalize();

            // Arrow shaft: from center outward
            let tip = (center + dir * arrow_length).normalize();

            // Color based on velocity intensity (heat map: blue=slow, red=fast)
            let max_speed = 1.0; // Approximate max velocity magnitude
            let t = (speed / max_speed).clamp(0.0, 1.0);
            // Interpolate hue from blue (240) to red (0)
            let hue = 240.0 * (1.0 - t);
            let color = hsl_to_rgb(hue, 0.9, 0.6);

            // Main shaft
            arrows.push((center, tip, color));

            // Arrowhead: two lines from tip going back at angles
            // Find perpendicular vector on the sphere surface
            let perp = center.cross(dir).normalize();

            // Arrowhead barbs
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
    pub fn generate_pole_markers(&self) -> Vec<(Vec3, Vec3, Vec3)> {
        let marker_size = 0.04; // Size of triangle
        let lift = 1.01; // Lift markers slightly off sphere to avoid Z-fighting
        let mut markers = Vec::new();

        for (plate_id, pole) in self.euler_poles.iter().enumerate() {
            let center = pole.axis.normalize();

            // Find two perpendicular vectors to the pole axis
            let up = if center.y.abs() < 0.9 {
                Vec3::Y
            } else {
                Vec3::X
            };
            let tangent1 = center.cross(up).normalize();
            let tangent2 = center.cross(tangent1).normalize();

            // Three points of the triangle, rotated 120 degrees apart
            let angle1 = 0.0_f32;
            let angle2 = std::f32::consts::TAU / 3.0;
            let angle3 = 2.0 * std::f32::consts::TAU / 3.0;

            // Calculate positions lifted off the surface
            let p1 = (center + (tangent1 * angle1.cos() + tangent2 * angle1.sin()) * marker_size)
                .normalize()
                * lift;
            let p2 = (center + (tangent1 * angle2.cos() + tangent2 * angle2.sin()) * marker_size)
                .normalize()
                * lift;
            let p3 = (center + (tangent1 * angle3.cos() + tangent2 * angle3.sin()) * marker_size)
                .normalize()
                * lift;

            // Normal points outward from sphere center (use triangle center direction)
            let normal = center;

            // Bright color for this plate
            let hue = (plate_id as f32 / self.num_plates as f32) * 360.0;
            let color = hsl_to_rgb(hue, 1.0, 0.8);

            markers.push((p1, normal, color));
            markers.push((p2, normal, color));
            markers.push((p3, normal, color));
        }

        markers
    }

    /// Generate boundary edge lines colored by stress.
    /// Returns line segments (start, end, color) for edges between different plates.
    pub fn generate_boundary_edges(&self, voronoi: &SphericalVoronoi) -> Vec<(Vec3, Vec3, Vec3)> {
        let mut result = Vec::new();
        let mut processed_edges: HashSet<(usize, usize)> = HashSet::new();
        let lift = 1.005; // Slight lift to render above cell edges

        // For each pair of adjacent cells on different plates, find their shared edge
        for (cell_idx, neighbors) in self.adjacency.iter().enumerate() {
            let plate_a = self.cell_plate[cell_idx] as usize;

            for &neighbor_idx in neighbors {
                let plate_b = self.cell_plate[neighbor_idx] as usize;

                // Only process boundary edges (different plates)
                if plate_a == plate_b {
                    continue;
                }

                // Canonical edge key to avoid processing twice
                let cell_pair = if cell_idx < neighbor_idx {
                    (cell_idx, neighbor_idx)
                } else {
                    (neighbor_idx, cell_idx)
                };

                if processed_edges.contains(&cell_pair) {
                    continue;
                }
                processed_edges.insert(cell_pair);

                // Find the shared edge (vertices that both cells have in common)
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

                let shared: Vec<usize> =
                    cell_verts.intersection(&neighbor_verts).copied().collect();

                if shared.len() != 2 {
                    continue; // Should always be 2 for adjacent Voronoi cells
                }

                let v0 = voronoi.vertices[shared[0]] * lift;
                let v1 = voronoi.vertices[shared[1]] * lift;

                // Calculate convergence for THIS SPECIFIC EDGE
                let cell_pos = voronoi.generators[cell_idx];
                let neighbor_pos = voronoi.generators[neighbor_idx];
                let boundary_point = (cell_pos + neighbor_pos).normalize();

                let vel_a = self.euler_poles[plate_a].velocity_at(boundary_point);
                let vel_b = self.euler_poles[plate_b].velocity_at(boundary_point);
                let relative_vel = vel_a - vel_b;

                // Boundary direction: project chord into tangent plane at boundary_point
                let chord = neighbor_pos - cell_pos;
                let tangent_normal = chord - boundary_point * chord.dot(boundary_point);
                let tangent_normal = if tangent_normal.length_squared() > 1e-10 {
                    tangent_normal.normalize()
                } else {
                    // Degenerate case: use arbitrary tangent
                    let up = if boundary_point.y.abs() < 0.9 {
                        Vec3::Y
                    } else {
                        Vec3::X
                    };
                    boundary_point.cross(up).normalize()
                };

                // If A moves toward B (relative_vel aligns with tangent_normal) → positive (convergent)
                // If A moves away from B → negative (divergent)
                let convergence = relative_vel.dot(tangent_normal);

                // Edge color shows motion type (convergent/divergent), not plate-type-specific stress
                let stress = convergence;

                // Color by stress - use smaller thresholds since these are individual edge values
                let color = if stress > 0.05 {
                    // Convergent - bright red
                    let t = (stress / 0.5).clamp(0.0, 1.0);
                    Vec3::new(0.8, 0.2, 0.1).lerp(Vec3::new(1.0, 0.3, 0.1), t)
                } else if stress < -0.05 {
                    // Divergent - bright blue
                    let t = (-stress / 0.3).clamp(0.0, 1.0);
                    Vec3::new(0.1, 0.3, 0.8).lerp(Vec3::new(0.2, 0.5, 1.0), t)
                } else {
                    // Neutral/transform - yellow
                    Vec3::new(0.8, 0.8, 0.2)
                };

                result.push((v0, v1, color));
            }
        }

        result
    }
}

/// Convert HSL to RGB.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Vec3 {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());

    let (r1, g1, b1) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    let m = l - c / 2.0;
    Vec3::new(r1 + m, g1 + m, b1 + m)
}

/// Find cells that are on plate boundaries (adjacent to cells on different plates).
fn find_boundary_cells(adjacency: &[Vec<usize>], cell_plate: &[u32]) -> Vec<usize> {
    let mut boundary = Vec::new();
    for (cell_idx, neighbors) in adjacency.iter().enumerate() {
        let my_plate = cell_plate[cell_idx];
        if neighbors.iter().any(|&n| cell_plate[n] != my_plate) {
            boundary.push(cell_idx);
        }
    }
    boundary
}

/// Build adjacency list: for each cell, list of neighboring cell indices.
///
/// Two cells are adjacent if they share an edge (two consecutive Voronoi vertices).
fn build_adjacency(voronoi: &SphericalVoronoi) -> Vec<Vec<usize>> {
    // Map from edge (as canonical vertex pair) to list of cells containing that edge
    let mut edge_to_cells: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

    for (cell_idx, cell) in voronoi.cells.iter().enumerate() {
        let verts = &cell.vertex_indices;
        let n = verts.len();

        for i in 0..n {
            let a = verts[i];
            let b = verts[(i + 1) % n];
            // Canonical ordering: smaller index first
            let edge = if a < b { (a, b) } else { (b, a) };
            edge_to_cells.entry(edge).or_default().push(cell_idx);
        }
    }

    // Build adjacency from edges shared by exactly 2 cells
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); voronoi.cells.len()];

    for cells in edge_to_cells.values() {
        if cells.len() == 2 {
            let c0 = cells[0];
            let c1 = cells[1];
            adjacency[c0].push(c1);
            adjacency[c1].push(c0);
        }
    }

    adjacency
}

/// Compute minimum angular distance between seeds based on plate count.
/// Uses a fraction of the ideal spacing (if seeds were perfectly distributed).
fn min_seed_distance(num_plates: usize) -> f32 {
    use super::tectonics::constants;
    // Ideal angular radius if N seeds perfectly tile the sphere
    let ideal = (1.0 - 2.0 / num_plates as f32).acos();
    ideal * constants::SEED_SPACING_FRACTION
}

/// Select plate seeds with minimum angular distance between them.
///
/// Uses rejection sampling: shuffle cells, accept each if it's far enough from
/// all previously accepted seeds. Min distance scales with plate count.
/// Falls back to progressively relaxed distances if needed.
fn select_seeds_spaced<R: Rng>(
    voronoi: &SphericalVoronoi,
    num_plates: usize,
    rng: &mut R,
) -> Vec<usize> {
    let base_min_dist = min_seed_distance(num_plates);

    let mut indices: Vec<usize> = (0..voronoi.cells.len()).collect();
    indices.shuffle(rng);

    let mut seeds: Vec<usize> = Vec::with_capacity(num_plates);
    let mut seed_positions: Vec<Vec3> = Vec::with_capacity(num_plates);

    // First pass: use full minimum distance
    for &cell_idx in &indices {
        if seeds.len() >= num_plates {
            break;
        }

        let pos = voronoi.generators[cell_idx];

        // Check distance from all existing seeds
        let far_enough = seed_positions
            .iter()
            .all(|&seed_pos| pos.dot(seed_pos).clamp(-1.0, 1.0).acos() >= base_min_dist);

        if far_enough {
            seeds.push(cell_idx);
            seed_positions.push(pos);
        }
    }

    // Fallback: progressively relax distance constraint until we have enough seeds
    let mut relaxation = 0.75;
    while seeds.len() < num_plates && relaxation > 0.0 {
        let relaxed_dist = base_min_dist * relaxation;
        indices.shuffle(rng);

        for &cell_idx in &indices {
            if seeds.len() >= num_plates {
                break;
            }

            // Skip if already a seed
            if seeds.contains(&cell_idx) {
                continue;
            }

            let pos = voronoi.generators[cell_idx];

            let far_enough = seed_positions
                .iter()
                .all(|&seed_pos| pos.dot(seed_pos).clamp(-1.0, 1.0).acos() >= relaxed_dist);

            if far_enough {
                seeds.push(cell_idx);
                seed_positions.push(pos);
            }
        }

        relaxation -= 0.25;
    }

    // Final fallback: if still short, add any remaining cells
    if seeds.len() < num_plates {
        for &cell_idx in &indices {
            if seeds.len() >= num_plates {
                break;
            }
            if !seeds.contains(&cell_idx) {
                seeds.push(cell_idx);
                seed_positions.push(voronoi.generators[cell_idx]);
            }
        }
    }

    seeds
}

/// Generate target sizes for plates using log-normal distribution.
///
/// Target size controls how many cells a plate will claim. The log-normal distribution
/// creates realistic size variation (like Earth's plates: Pacific huge, Juan de Fuca tiny).
fn generate_target_sizes<R: Rng>(num_plates: usize, rng: &mut R) -> Vec<f32> {
    use super::tectonics::constants;

    let normal = Normal::new(0.0, constants::TARGET_SIZE_SIGMA as f64).unwrap();

    // Generate raw log-normal values
    let mut sizes: Vec<f32> = (0..num_plates)
        .map(|_| (normal.sample(rng) as f32).exp())
        .collect();

    // Clamp ratio between max and min
    let max_val = sizes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_val = sizes.iter().cloned().fold(f32::INFINITY, f32::min);

    if max_val > min_val * constants::TARGET_SIZE_MAX_RATIO {
        // Scale up small values to fit within ratio
        let target_min = max_val / constants::TARGET_SIZE_MAX_RATIO;
        for s in &mut sizes {
            if *s < target_min {
                *s = target_min;
            }
        }
    }

    // Normalize so sum = num_plates (makes fill_ratio intuitive)
    let sum: f32 = sizes.iter().sum();
    let scale = num_plates as f32 / sum;
    for s in &mut sizes {
        *s *= scale;
    }

    sizes
}

/// Per-plate state for flood fill.
struct PlateState {
    seed_pos: Vec3,
    target_size: f32,
    noise_offset: Vec3,
}

impl PlateState {
    fn new<R: Rng>(seed_pos: Vec3, target_size: f32, rng: &mut R) -> Self {
        let noise_offset = Vec3::new(
            rng.gen::<f32>() * 100.0,
            rng.gen::<f32>() * 100.0,
            rng.gen::<f32>() * 100.0,
        );
        PlateState {
            seed_pos,
            target_size,
            noise_offset,
        }
    }
}

/// Calculate priority for a cell (lower = claim sooner).
fn compute_priority(
    cell_pos: Vec3,
    plate: &PlateState,
    total_neighbors: usize,
    same_plate_neighbors: usize,
    fbm: &Fbm<Perlin>,
) -> f32 {
    use super::tectonics::constants;

    // Arc distance from seed, scaled by target_size
    // Larger plates see smaller effective distances, so they grow larger
    let distance = plate.seed_pos.dot(cell_pos).clamp(-1.0, 1.0).acos();
    let scaled_distance = distance / plate.target_size;

    // Noise at this position (with plate-specific offset)
    let noise_pos = cell_pos * constants::NOISE_FREQUENCY as f32 + plate.noise_offset;
    let noise_val = fbm.get([noise_pos.x as f64, noise_pos.y as f64, noise_pos.z as f64]) as f32;

    // Net perimeter change: lower = better (less perimeter added)
    let perimeter_delta = total_neighbors as f32 - 2.0 * same_plate_neighbors as f32;

    scaled_distance
        + constants::NOISE_WEIGHT * noise_val * 0.5
        + constants::NEIGHBOR_BONUS * perimeter_delta
}

/// Weighted flood fill with single global priority queue.
///
/// Algorithm:
/// 1. Single global queue of (priority, cell_idx, plate_id)
/// 2. Pop globally best cell, claim it for associated plate
/// 3. Add unclaimed neighbors to global queue
/// 4. Distance naturally causes plates to alternate
fn flood_fill_weighted<R: Rng>(
    adjacency: &[Vec<usize>],
    seeds: &[usize],
    target_sizes: &[f32],
    voronoi: &SphericalVoronoi,
    rng: &mut R,
) -> Vec<u32> {
    use super::tectonics::constants;

    let num_cells = voronoi.cells.len();
    let mut cell_plate = vec![u32::MAX; num_cells]; // Unassigned

    // Create fBm noise generator
    let fbm: Fbm<Perlin> = Fbm::new(rng.gen())
        .set_frequency(constants::NOISE_FREQUENCY)
        .set_octaves(constants::NOISE_OCTAVES);

    // Initialize plate states
    let plates: Vec<PlateState> = seeds
        .iter()
        .enumerate()
        .map(|(plate_id, &seed_cell)| {
            let seed_pos = voronoi.generators[seed_cell];
            PlateState::new(seed_pos, target_sizes[plate_id], rng)
        })
        .collect();

    // Single global priority queue: (priority, cell_idx, plate_id)
    let mut queue: BinaryHeap<Reverse<(OrderedFloat<f32>, usize, usize)>> = BinaryHeap::new();

    // Initialize: claim seeds and add their neighbors to queue
    for (plate_id, &seed_cell) in seeds.iter().enumerate() {
        cell_plate[seed_cell] = plate_id as u32;

        for &neighbor in &adjacency[seed_cell] {
            if cell_plate[neighbor] == u32::MAX {
                let neighbor_pos = voronoi.generators[neighbor];
                let total_neighbors = adjacency[neighbor].len();
                let same_plate_neighbors = adjacency[neighbor]
                    .iter()
                    .filter(|&&n| cell_plate[n] == plate_id as u32)
                    .count();
                let priority = compute_priority(
                    neighbor_pos,
                    &plates[plate_id],
                    total_neighbors,
                    same_plate_neighbors,
                    &fbm,
                );
                queue.push(Reverse((OrderedFloat(priority), neighbor, plate_id)));
            }
        }
    }

    // Main loop: pop globally best cell
    while let Some(Reverse((_, cell_idx, plate_id))) = queue.pop() {
        if cell_plate[cell_idx] != u32::MAX {
            continue; // Already claimed
        }

        // Claim cell
        cell_plate[cell_idx] = plate_id as u32;

        // Add unclaimed neighbors to global queue
        for &neighbor in &adjacency[cell_idx] {
            if cell_plate[neighbor] == u32::MAX {
                let neighbor_pos = voronoi.generators[neighbor];
                let total_neighbors = adjacency[neighbor].len();
                let same_plate_neighbors = adjacency[neighbor]
                    .iter()
                    .filter(|&&n| cell_plate[n] == plate_id as u32)
                    .count();
                let priority = compute_priority(
                    neighbor_pos,
                    &plates[plate_id],
                    total_neighbors,
                    same_plate_neighbors,
                    &fbm,
                );
                queue.push(Reverse((OrderedFloat(priority), neighbor, plate_id)));
            }
        }
    }

    cell_plate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{random_sphere_points, SphericalVoronoi};

    #[test]
    fn test_plate_generation() {
        let points = random_sphere_points(100);
        let voronoi = SphericalVoronoi::compute(&points);
        let plates = TectonicPlates::generate(&voronoi, 5);

        // All cells assigned to a valid plate
        assert!(plates.cell_plate.iter().all(|&p| p < 5));
        // Correct number of plate types
        assert_eq!(plates.plate_types.len(), 5);
        // Correct number of Euler poles
        assert_eq!(plates.euler_poles.len(), 5);
        // Correct number of cells
        assert_eq!(plates.cell_plate.len(), voronoi.cells.len());
        // Correct number of elevations
        assert_eq!(plates.cell_elevation.len(), voronoi.cells.len());
    }

    #[test]
    fn test_adjacency_symmetric() {
        let points = random_sphere_points(50);
        let voronoi = SphericalVoronoi::compute(&points);
        let adjacency = build_adjacency(&voronoi);

        // Adjacency should be symmetric
        for (cell, neighbors) in adjacency.iter().enumerate() {
            for &neighbor in neighbors {
                assert!(
                    adjacency[neighbor].contains(&cell),
                    "Adjacency not symmetric: {} -> {} but not reverse",
                    cell,
                    neighbor
                );
            }
        }
    }
}
