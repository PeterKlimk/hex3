//! World generation and simulation.
//!
//! This module contains the domain model for procedural world generation,
//! organized into progressive stages:
//!
//! **Stage 1: Lithosphere (Tectonics)**
//! - Tessellation - Spherical Voronoi cells + adjacency graph
//! - Plates - Tectonic plate assignment via flood fill
//! - Dynamics - Plate motion (Euler poles) and types (continental/oceanic)
//! - Stress - Boundary stress calculation and propagation
//! - Elevation - Terrain height from stress + noise
//!
//! **Stage 2: Hydrosphere**
//! - Hydrology - Depression filling, drainage, rivers
//!
//! Future stages: Climate, Biomes

mod constants;
mod boundary;
mod dynamics;
mod elevation;
mod hydrology;
mod plates;
mod stress;
mod tessellation;

pub mod gen;

pub use constants::*;
pub use boundary::{collect_plate_boundaries, BoundaryKind, PlateBoundaryEdge, SubductionPolarity};
pub use dynamics::{Dynamics, EulerPole, PlateType};
pub use elevation::{Elevation, NoiseLayerData};
pub use hydrology::{Basin, CellWaterState, Hydrology, WaterBody, DEFAULT_CLIMATE_RATIO};
pub use plates::Plates;
pub use stress::StressField;
pub use tessellation::Tessellation;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A procedurally generated world with layered geological features.
///
/// The world is built up in stages, each depending on previous stages.
/// Stages can be generated all at once or incrementally for visualization.
pub struct World {
    /// Random seed used for reproducible generation.
    pub seed: u64,

    /// Base tessellation - always present after construction.
    pub tessellation: Tessellation,

    // --- Stage 1: Lithosphere ---
    /// Tectonic plate assignments.
    pub plates: Option<Plates>,

    /// Plate dynamics (motion and types).
    pub dynamics: Option<Dynamics>,

    /// Stress field from plate interactions.
    pub stress: Option<StressField>,

    /// Terrain elevation.
    pub elevation: Option<Elevation>,

    // --- Stage 2: Hydrosphere ---
    /// Hydrology (drainage, rivers).
    pub hydrology: Option<Hydrology>,
}

impl World {
    /// Create a new world with the given seed and number of cells.
    ///
    /// This only generates the tessellation. Call generation methods
    /// to build up additional layers.
    pub fn new(seed: u64, num_cells: usize, lloyd_iterations: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tessellation = Tessellation::generate(num_cells, lloyd_iterations, &mut rng);

        Self {
            seed,
            tessellation,
            plates: None,
            dynamics: None,
            stress: None,
            elevation: None,
            hydrology: None,
        }
    }

    /// Generate all stages at once with the given parameters.
    pub fn generate_all(&mut self, num_plates: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        // Skip the tessellation RNG usage to stay in sync
        let _ = Tessellation::generate(
            self.tessellation.num_cells(),
            0, // doesn't matter, we're just advancing RNG
            &mut rng,
        );

        self.generate_plates(num_plates);
        self.generate_dynamics();
        self.generate_stress();
        self.generate_elevation();
    }

    /// Generate tectonic plates via flood fill.
    pub fn generate_plates(&mut self, num_plates: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(1));
        self.plates = Some(Plates::generate(
            &self.tessellation,
            num_plates,
            &mut rng,
        ));
    }

    /// Generate plate dynamics (Euler poles and types).
    /// Requires plates to be generated first.
    pub fn generate_dynamics(&mut self) {
        let plates = self.plates.as_ref().expect("Plates must be generated first");
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(2));
        self.dynamics = Some(Dynamics::generate(plates, &mut rng));
    }

    /// Generate stress field from plate interactions.
    /// Requires plates and dynamics to be generated first.
    pub fn generate_stress(&mut self) {
        let plates = self.plates.as_ref().expect("Plates must be generated first");
        let dynamics = self.dynamics.as_ref().expect("Dynamics must be generated first");
        self.stress = Some(StressField::calculate(
            &self.tessellation,
            plates,
            dynamics,
        ));
    }

    /// Generate elevation from stress.
    /// Requires stress and dynamics to be generated first.
    pub fn generate_elevation(&mut self) {
        let plates = self.plates.as_ref().expect("Plates must be generated first");
        let dynamics = self.dynamics.as_ref().expect("Dynamics must be generated first");
        let stress = self.stress.as_ref().expect("Stress must be generated first");
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(3));
        self.elevation = Some(Elevation::generate(
            &self.tessellation,
            plates,
            dynamics,
            stress,
            &mut rng,
        ));
    }

    /// Generate hydrology (drainage, rivers).
    /// Requires plates, dynamics, and elevation to be generated first.
    pub fn generate_hydrology(&mut self) {
        let plates = self.plates.as_ref().expect("Plates must be generated first");
        let dynamics = self
            .dynamics
            .as_ref()
            .expect("Dynamics must be generated first");
        let elevation = self
            .elevation
            .as_ref()
            .expect("Elevation must be generated first");
        self.hydrology = Some(Hydrology::generate(
            &self.tessellation,
            plates,
            dynamics,
            elevation,
        ));
    }

    /// Get the number of cells in this world.
    pub fn num_cells(&self) -> usize {
        self.tessellation.num_cells()
    }

    /// Get the current generation stage.
    pub fn current_stage(&self) -> u32 {
        if self.hydrology.is_some() {
            2
        } else if self.elevation.is_some() {
            1
        } else {
            0
        }
    }
}
