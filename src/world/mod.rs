//! World generation and simulation.
//!
//! This module contains the domain model for procedural world generation,
//! organized into progressive stages:
//!
//! **Stage 1: Lithosphere (Tectonics)**
//! - Tessellation - Spherical Voronoi cells + adjacency graph
//! - Plates - Tectonic plate assignment via flood fill (motion units)
//! - Crust - Per-cell continental/oceanic crust, grown independently of plates
//! - Dynamics - Plate motion (Euler poles)
//! - Features - Boundary-driven terrain features (trenches, arcs, ridges, collision zones)
//! - Elevation - Terrain height from features + noise
//!
//! **Stage 2: Atmosphere (Climate)**
//! - Climate - Temperature, pressure, wind fields
//! - Moisture - Advected moisture and precipitation
//!
//! **Stage 3: Hydrosphere**
//! - Hydrology - Depression filling, drainage, rivers
//!
//! Future stages: Erosion, Biomes

mod atmosphere;
mod boundary;
mod constants;
mod crust;
pub mod diagnostics;
mod dynamics;
mod elevation;
mod features;
mod hydrology;
mod moisture;
mod plates;
mod tessellation;

pub use atmosphere::Atmosphere;
pub use boundary::{collect_plate_boundaries, BoundaryKind, PlateBoundaryEdge, SubductionPolarity};
pub use constants::*;
pub use crust::{Crust, CrustType};

/// Default plate count used by the app and diagnostic tooling.
pub const NUM_PLATES_DEFAULT: usize = 14;
pub use dynamics::{Dynamics, EulerPole};
pub use elevation::{Elevation, NoiseLayerData};
pub use features::FeatureFields;
pub use hydrology::{Basin, CellWaterState, Hydrology, WaterBody, DEFAULT_CLIMATE_RATIO};
pub use plates::Plates;
pub use tessellation::Tessellation;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::fmt;

/// Backend used to compute the spherical Voronoi diagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoronoiBackend {
    /// Exact convex-hull duality (slower, robust).
    #[default]
    ConvexHull,
    /// kNN-driven half-space clipping (fast, approximate).
    KnnClipping,
}

impl fmt::Display for VoronoiBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VoronoiBackend::ConvexHull => write!(f, "convex-hull"),
            VoronoiBackend::KnnClipping => write!(f, "knn-clipping"),
        }
    }
}

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

    /// Per-cell crust type (continental vs oceanic), independent of plates.
    pub crust: Option<Crust>,

    /// Plate dynamics (motion).
    pub dynamics: Option<Dynamics>,

    /// Tectonic feature fields (trench, arc, ridge, collision, activity).
    pub features: Option<FeatureFields>,

    /// Terrain elevation.
    pub elevation: Option<Elevation>,

    // --- Stage 2: Atmosphere ---
    /// Atmosphere data (temperature, pressure, wind, uplift).
    pub atmosphere: Option<Atmosphere>,

    // --- Stage 3: Hydrosphere ---
    /// Hydrology (drainage, rivers).
    pub hydrology: Option<Hydrology>,
}

impl World {
    /// Create a new world with the given seed and number of cells.
    ///
    /// This only generates the tessellation. Call generation methods
    /// to build up additional layers.
    pub fn new(seed: u64, num_cells: usize, lloyd_iterations: usize) -> Self {
        Self::new_with_options(
            seed,
            num_cells,
            lloyd_iterations,
            VoronoiBackend::ConvexHull,
        )
    }

    /// Create a new world with options.
    ///
    /// Selects the Voronoi backend for tessellation.
    pub fn new_with_options(
        seed: u64,
        num_cells: usize,
        lloyd_iterations: usize,
        backend: VoronoiBackend,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let tessellation = match backend {
            VoronoiBackend::ConvexHull => {
                Tessellation::generate(num_cells, lloyd_iterations, &mut rng)
            }
            VoronoiBackend::KnnClipping => {
                Tessellation::generate_knn_clipping(num_cells, lloyd_iterations, &mut rng)
            }
        };

        Self {
            seed,
            tessellation,
            plates: None,
            crust: None,
            dynamics: None,
            features: None,
            elevation: None,
            atmosphere: None,
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
        self.generate_crust();
        self.generate_dynamics();
        self.generate_features();
        self.generate_elevation();
    }

    /// Generate tectonic plates via flood fill.
    pub fn generate_plates(&mut self, num_plates: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(1));
        self.plates = Some(Plates::generate(&self.tessellation, num_plates, &mut rng));
    }

    /// Generate per-cell crust (continents grown independently of plates).
    pub fn generate_crust(&mut self) {
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(4));
        self.crust = Some(Crust::generate(
            &self.tessellation,
            NUM_CRATONS,
            CONTINENTAL_FRACTION,
            &mut rng,
        ));
    }

    /// Generate plate dynamics (Euler poles).
    /// Requires plates to be generated first.
    pub fn generate_dynamics(&mut self) {
        let plates = self
            .plates
            .as_ref()
            .expect("Plates must be generated first");
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(2));
        self.dynamics = Some(Dynamics::generate(plates, &mut rng));
    }

    /// Generate tectonic feature fields (trench, arc, ridge, collision, activity).
    /// Requires plates, crust, and dynamics to be generated first.
    pub fn generate_features(&mut self) {
        let plates = self
            .plates
            .as_ref()
            .expect("Plates must be generated first");
        let crust = self.crust.as_ref().expect("Crust must be generated first");
        let dynamics = self
            .dynamics
            .as_ref()
            .expect("Dynamics must be generated first");
        self.features = Some(FeatureFields::compute(
            &self.tessellation,
            plates,
            crust,
            dynamics,
        ));
    }

    /// Generate elevation from tectonic features.
    /// Requires crust and features to be generated first.
    pub fn generate_elevation(&mut self) {
        let crust = self.crust.as_ref().expect("Crust must be generated first");
        let features = self
            .features
            .as_ref()
            .expect("Features must be generated first");
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(3));
        self.elevation = Some(Elevation::generate(
            &self.tessellation,
            crust,
            features,
            &mut rng,
        ));
    }

    /// Generate atmosphere (temperature, pressure, wind, uplift).
    /// Requires elevation to be generated first.
    pub fn generate_atmosphere(&mut self) {
        let elevation = self
            .elevation
            .as_ref()
            .expect("Elevation must be generated first");
        self.atmosphere = Some(Atmosphere::generate(&self.tessellation, elevation));
    }

    /// Generate hydrology (drainage, rivers).
    /// Requires crust, elevation, and atmosphere (for precipitation).
    pub fn generate_hydrology(&mut self) {
        let crust = self.crust.as_ref().expect("Crust must be generated first");
        let elevation = self
            .elevation
            .as_ref()
            .expect("Elevation must be generated first");
        let atmosphere = self
            .atmosphere
            .as_ref()
            .expect("Atmosphere must be generated first");
        self.hydrology = Some(Hydrology::generate(
            &self.tessellation,
            crust,
            elevation,
            &atmosphere.precipitation,
            &atmosphere.temperature,
        ));
    }

    /// Get the number of cells in this world.
    pub fn num_cells(&self) -> usize {
        self.tessellation.num_cells()
    }

    /// Get the current generation stage.
    /// - Stage 1: Lithosphere (tectonics, elevation)
    /// - Stage 2: Atmosphere (temperature, wind)
    /// - Stage 3: Hydrosphere (rivers, lakes)
    pub fn current_stage(&self) -> u32 {
        if self.hydrology.is_some() {
            3
        } else if self.atmosphere.is_some() {
            2
        } else if self.elevation.is_some() {
            1
        } else {
            0
        }
    }
}
