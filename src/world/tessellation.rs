//! Spherical tessellation - Voronoi cells and adjacency graph.

use std::collections::HashMap;

use glam::Vec3;
use rand::Rng;

use crate::geometry::{lloyd_relax, random_sphere_points_with_rng, SphericalVoronoi};

/// A spherical tessellation with Voronoi cells and cell adjacency.
pub struct Tessellation {
    /// The underlying Voronoi diagram.
    pub voronoi: SphericalVoronoi,

    /// Adjacency list: for each cell, indices of neighboring cells.
    pub adjacency: Vec<Vec<usize>>,
}

impl Tessellation {
    /// Generate a tessellation with the given number of cells.
    pub fn generate<R: Rng>(num_cells: usize, lloyd_iterations: usize, rng: &mut R) -> Self {
        let mut points = random_sphere_points_with_rng(num_cells, rng);
        lloyd_relax(&mut points, lloyd_iterations);
        let voronoi = SphericalVoronoi::compute(&points);
        let adjacency = build_adjacency(&voronoi);

        Self { voronoi, adjacency }
    }

    /// Number of cells in this tessellation.
    pub fn num_cells(&self) -> usize {
        self.voronoi.cells.len()
    }

    /// Get the center point (generator) of a cell.
    pub fn cell_center(&self, cell_idx: usize) -> Vec3 {
        self.voronoi.generators[cell_idx]
    }

    /// Get the neighbors of a cell.
    pub fn neighbors(&self, cell_idx: usize) -> &[usize] {
        &self.adjacency[cell_idx]
    }
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
