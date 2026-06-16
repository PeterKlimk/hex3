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
mod circulation;
mod constants;
mod crust;
pub mod diagnostics;
mod dynamics;
mod elevation;
mod erosion;
mod features;
mod fine;
mod fine_cache;
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
pub use erosion::{roughness_counters, ErosionParams, RoughnessCounters};
pub use features::FeatureFields;
pub use fine::{FineBase, FineDensityParams, FineFields, FineSurface, FineWorld};
pub use fine_cache::FineCacheMode;
pub use hydrology::{Basin, CellWaterState, Hydrology, WaterBody, DEFAULT_CLIMATE_RATIO};
pub use plates::Plates;
pub use tessellation::{CellAdjacency, Tessellation};

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

    /// Adaptive fine mesh and Stage-3 hydrology/erosion infrastructure.
    pub fine: Option<FineWorld>,

    /// Runtime-tunable erosion knobs (defaults from `EROSION_*` consts). Used
    /// when (re-)generating the fine-mesh erosion surface.
    pub erosion_params: ErosionParams,

    /// Runtime-tunable fine-mesh density knobs (defaults from `FINE_*` consts).
    /// Used when (re-)generating the fine-mesh base; part of its cache key.
    pub fine_density_params: FineDensityParams,

    /// Whether fine-mesh base generation reads/writes the on-disk cache.
    pub fine_cache: FineCacheMode,

    /// Transient view cap for stage back-navigation: the `active_*` accessors and
    /// fine-mesh rendering only expose data up to this stage, so the app can
    /// render an earlier (already-computed) stage without recompute. Defaults to
    /// `u32::MAX` (uncapped) so headless/batch paths behave exactly as before.
    view_stage: u32,
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
            fine: None,
            erosion_params: ErosionParams::default(),
            fine_density_params: FineDensityParams::default(),
            fine_cache: FineCacheMode::default(),
            view_stage: u32::MAX,
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

    /// Generate the full stage-4 world (fine base + pre-erosion + eroded) at the
    /// default cap. Used by headless/batch + export; the interactive app builds
    /// stage 3 (pre) and stage 4 (eroded) separately so it can snap between them.
    pub fn generate_hydrology(&mut self) {
        self.generate_hydrology_with_fine_cap(FINE_MAX_CELLS);
    }

    /// Full stage-4 generation with an explicit fine-mesh cell cap (diagnostics
    /// pass a smaller cap to iterate on a proportionally-coarsened mesh).
    pub fn generate_hydrology_with_fine_cap(&mut self, fine_max_cells: usize) {
        self.generate_fine_pre_with_cap(fine_max_cells);
        self.generate_fine_eroded();
    }

    /// Stage 3 (Hydrosphere): build the fine base (cache-aware) and the
    /// pre-erosion surface (hydrology on the un-eroded base). Erosion is stage 4.
    pub fn generate_fine_pre(&mut self) {
        self.generate_fine_pre_with_cap(FINE_MAX_CELLS);
    }

    /// Stage 3 with an explicit fine-mesh cell cap.
    pub fn generate_fine_pre_with_cap(&mut self, fine_max_cells: usize) {
        let crust = self.crust.as_ref().expect("Crust must be generated first");
        let features = self
            .features
            .as_ref()
            .expect("Features must be generated first");
        let elevation = self
            .elevation
            .as_ref()
            .expect("Elevation must be generated first");
        let atmosphere = self
            .atmosphere
            .as_ref()
            .expect("Atmosphere must be generated first");
        let fine = FineWorld::generate_pre(
            self.seed,
            &self.tessellation,
            crust,
            features,
            elevation,
            atmosphere,
            fine_max_cells,
            self.fine_cache,
            self.fine_density_params,
        );
        self.hydrology = None;
        self.fine = Some(fine);
    }

    /// Stage 4 (Erosion): compute the eroded surface over the existing fine base
    /// (no-op if stage 3 isn't built or erosion is already computed).
    pub fn generate_fine_eroded(&mut self) {
        let seed = self.seed;
        let params = self.erosion_params;
        if let Some(fine) = self.fine.as_mut() {
            fine.compute_eroded(seed, params);
        }
    }

    /// Re-run the eroded surface with the current `erosion_params` (e.g. after a
    /// runtime knob change), replacing it. Reuses the base — no mesh recompute.
    pub fn rerun_fine_eroded(&mut self) {
        let seed = self.seed;
        let params = self.erosion_params;
        if let Some(fine) = self.fine.as_mut() {
            fine.rerun_eroded(seed, params);
        }
    }

    /// Get the number of cells in this world.
    pub fn num_cells(&self) -> usize {
        self.active_tessellation().num_cells()
    }

    /// Cap the view at `stage` for back-navigation (the `active_*` accessors and
    /// fine rendering expose data only up to it). `u32::MAX` = uncapped.
    pub fn set_view_stage(&mut self, stage: u32) {
        self.view_stage = stage;
    }

    /// Current view cap.
    pub fn view_stage(&self) -> u32 {
        self.view_stage
    }

    /// Whether fine-mesh data should be shown (fine exists AND the view isn't
    /// capped below stage 3). Buffer generation must use this — NOT
    /// `fine.is_some()` — so back-navigation renders the coarse mesh consistently.
    pub fn shows_fine(&self) -> bool {
        self.view_stage >= 3 && self.fine.is_some()
    }

    pub fn active_tessellation(&self) -> &Tessellation {
        if self.shows_fine() {
            self.fine.as_ref().unwrap().tessellation()
        } else {
            &self.tessellation
        }
    }

    pub fn active_elevation(&self) -> Option<&Elevation> {
        if self.shows_fine() {
            // Stage 3 -> pre-erosion surface; stage 4 -> eroded surface.
            return Some(
                &self
                    .fine
                    .as_ref()
                    .unwrap()
                    .surface_for(self.view_stage)
                    .elevation,
            );
        }
        // Elevation is stage 1, shown at any view stage >= 1.
        self.elevation.as_ref()
    }

    pub fn active_hydrology(&self) -> Option<&Hydrology> {
        if self.view_stage < 3 {
            return None; // hydrology is a stage-3 layer
        }
        if let Some(fine) = self.fine.as_ref() {
            return Some(&fine.surface_for(self.view_stage).hydrology);
        }
        self.hydrology.as_ref()
    }

    pub fn active_hydrology_mut(&mut self) -> Option<&mut Hydrology> {
        if self.view_stage < 3 {
            return None;
        }
        let vs = self.view_stage;
        if let Some(fine) = self.fine.as_mut() {
            let surface = if vs >= 4 && fine.eroded.is_some() {
                fine.eroded.as_mut().unwrap()
            } else {
                &mut fine.pre
            };
            return Some(&mut surface.hydrology);
        }
        self.hydrology.as_mut()
    }

    pub fn active_temperature(&self) -> Option<&[f32]> {
        if self.view_stage < 2 {
            return None; // atmosphere is a stage-2 layer
        }
        if self.shows_fine() {
            return Some(self.fine.as_ref().unwrap().fields().temperature.as_slice());
        }
        self.atmosphere.as_ref().map(|a| a.temperature.as_slice())
    }

    pub fn active_precipitation(&self) -> Option<&[f32]> {
        if self.view_stage < 2 {
            return None;
        }
        if self.shows_fine() {
            return Some(
                self.fine
                    .as_ref()
                    .unwrap()
                    .surface_for(self.view_stage)
                    .precipitation
                    .as_slice(),
            );
        }
        self.atmosphere.as_ref().map(|a| a.precipitation.as_slice())
    }

    pub fn active_uplift(&self) -> Option<&[f32]> {
        if self.view_stage < 2 {
            return None;
        }
        if self.shows_fine() {
            return Some(self.fine.as_ref().unwrap().fields().uplift.as_slice());
        }
        self.atmosphere.as_ref().map(|a| a.uplift.as_slice())
    }

    pub fn set_active_climate_ratio(&mut self, ratio: f32) {
        let vs = self.view_stage;
        if let Some(fine) = &mut self.fine {
            fine.set_climate_ratio(vs, ratio);
        } else if let Some(hydrology) = &mut self.hydrology {
            hydrology.set_climate_ratio(&self.tessellation, ratio);
        }
    }

    /// Get the current (max computed) generation stage.
    /// - Stage 1: Lithosphere (tectonics, elevation)
    /// - Stage 2: Atmosphere (temperature, wind)
    /// - Stage 3: Hydrosphere (fine mesh, rivers on pre-erosion terrain)
    /// - Stage 4: Erosion (eroded terrain + re-derived rivers)
    pub fn current_stage(&self) -> u32 {
        if let Some(fine) = &self.fine {
            return if fine.has_eroded() { 4 } else { 3 };
        }
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
