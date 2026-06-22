//! Hydrology simulation - basins, water levels, rivers, and drainage.
//!
//! Algorithm:
//! 1. Smooth elevation to reduce noise-induced micro-depressions
//! 2. Identify ocean cells (connected below-sea-level regions touching oceanic crust)
//! 3. Priority-flood from ocean to detect all basins (depressions)
//! 4. Compute drainage directions and flow accumulation
//! 5. Calculate catchment area for each basin
//! 6. Determine equilibrium water level based on catchment vs surface area

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::time::Instant;

use ordered_float::OrderedFloat;

use super::constants::EVAP_TEMP_SENSITIVITY;
use super::{Crust, Elevation, Tessellation};

/// Water state of a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellWaterState {
    /// Regular land - not in ocean or any basin.
    Land,
    /// Ocean water.
    Ocean,
    /// Lake water - in a basin, below the current water level.
    LakeWater,
    /// Dry basin - in a basin, but above the current water level (lake bed, playa).
    DryBasin,
}

/// A topographic basin (depression) that can hold water.
///
/// The basin includes ALL cells below the spill point, regardless of
/// whether they're currently underwater. The water_level determines
/// which cells are actually submerged.
#[derive(Debug, Clone)]
pub struct Basin {
    /// All cells in this basin (below spill elevation).
    pub cells: Vec<usize>,
    /// Cell elevations sorted ascending (for hypsometric lookup).
    /// sorted_elevations[i] is the elevation of the i-th lowest cell.
    pub sorted_elevations: Vec<f32>,
    /// Cell areas (steradians) aligned with `sorted_elevations`: sorted_areas[i]
    /// is the area of the i-th lowest cell. Lets the hypsometric fill measure the
    /// submerged surface by AREA, not cell count — essential on the adaptive fine
    /// mesh where basin cells span ~50:1 in area.
    pub sorted_areas: Vec<f32>,
    /// Total basin surface area (steradians) = sum of `sorted_areas`. The lake's
    /// maximum (spill-full) surface area.
    pub total_area: f32,
    /// Spill elevation - maximum possible water level.
    pub spill_elevation: f32,
    /// Bottom elevation - lowest point in the basin.
    pub bottom_elevation: f32,
    /// The cell just outside the basin where overflow exits.
    /// Following drainage from here leads to ocean or another basin.
    pub spill_target_cell: usize,
    /// Which basin this one overflows into (None = drains to ocean).
    pub overflow_target: Option<usize>,
    /// Upstream discharge draining into this basin: precipitation-weighted
    /// drainage AREA (units: rainfall × steradian), so it shares erosion's
    /// "wet area" definition and is correct on the adaptive mesh.
    pub catchment_area: f32,
    /// Area-weighted mean surface temperature of the basin catchment.
    pub mean_temperature: f32,
    /// Evaporation multiplier derived from mean catchment temperature.
    /// 1.0 means the global climate ratio behaves exactly as before.
    pub evaporation_factor: f32,
    /// Current water surface elevation.
    /// If < bottom_elevation, the basin is dry.
    /// If >= spill_elevation, the basin is full/overflowing.
    pub water_level: f32,
}

impl Basin {
    /// Check if this basin is currently overflowing (full to spill point).
    pub fn is_overflowing(&self) -> bool {
        self.water_level >= self.spill_elevation
    }

    /// Check if this basin has water (not dry).
    pub fn has_water(&self) -> bool {
        self.water_level > self.bottom_elevation
    }
}

/// A connected component of underwater cells that forms a distinct water body.
///
/// IMPORTANT: Water bodies are recomputed when climate changes. Indices are
/// NOT stable across climate adjustments - lakes can split, merge, appear,
/// or disappear. For stable references to named features, use basin_id +
/// spatial anchoring (e.g., deepest cell) instead.
///
/// TODO: Consider adding a Wetland classification for shallow/small water
/// bodies that shouldn't act as river sinks but might affect rendering.
#[derive(Debug, Clone)]
pub struct WaterBody {
    /// Cells in this water body (connected component of underwater cells).
    pub cells: Vec<usize>,
    /// Which basin this water body is in.
    pub basin_id: usize,
    /// Maximum depth (water_level - lowest cell elevation).
    pub max_depth: f32,
    /// Whether this qualifies as a proper lake (passes depth threshold).
    pub is_lake: bool,
}

/// Hydrology data for the world.
pub struct Hydrology {
    /// Elevation used for hydrology calculations.
    pub elevation: Vec<f32>,

    /// Depression-filled elevation (every cell can drain to ocean).
    pub filled_elevation: Vec<f32>,

    /// For each cell, which neighbor does water flow to?
    /// None means this cell is water (ocean/lake) or has no valid outlet.
    pub drainage_dir: Vec<Option<usize>>,

    /// Accumulated flow - how many upstream cells drain through this one.
    /// Higher values = larger rivers.
    pub flow_accumulation: Vec<f32>,

    /// Whether each cell is ocean.
    pub is_ocean: Vec<bool>,

    /// Which basin (if any) each cell belongs to.
    /// This includes ALL cells in the depression, not just underwater ones.
    pub basin_id: Vec<Option<usize>>,

    /// All basins (topographic depressions). Basins are STATIC - defined by
    /// terrain topology, not current water levels.
    pub basins: Vec<Basin>,

    /// Water bodies derived from current water levels (connected wet components).
    /// DYNAMIC - recomputed when climate changes. Indices are not stable.
    pub water_bodies: Vec<WaterBody>,

    /// Which water body (if any) each cell belongs to.
    /// Only set for cells that are in a lake (pass depth threshold).
    pub cell_water_body: Vec<Option<usize>>,

    /// Current climate ratio (precipitation / evaporation).
    /// Controls how full lakes are. Higher = wetter = more/larger lakes.
    pub climate_ratio: f32,

    /// Precipitation-weighted mean cell area, Σ(precip·area) / Σ(precip)
    /// (steradians). `flow_accumulation` is a physical discharge (precip ×
    /// steradian); dividing by this converts it back to a count-equivalent
    /// (≈ the old precip-weighted upstream cell count on the equal-area coarse
    /// mesh), so legacy count-based thresholds (river density, density prior,
    /// coloring) keep their meaning and become resolution-independent.
    pub mean_cell_discharge: f32,
}

/// Minimum ocean component size as fraction of total cells.
const MIN_OCEAN_AREA_FRACTION: f32 = 0.001;

/// Default climate ratio (precipitation / evaporation).
/// Values > 1.0 = wet climate (more lakes)
/// Values < 1.0 = arid climate (fewer lakes)
/// At equilibrium: lake_surface_area = catchment_area × climate_ratio
pub const DEFAULT_CLIMATE_RATIO: f32 = 0.15;

// --- Drainage integration (docs/specs/drainage-integration.md) ---
// Real drainage networks INTEGRATE over geologic time: rivers breach divides and basins
// capture each other, so most interior basins have an inherited outlet even in arid present
// climates. The generator only does depression FILLING, leaving ~30-50% of land endorheic.
// This carves outlet channels (along the priority-flood spill path) for basins that geologic
// time would have integrated, BEFORE the present-climate hydrology runs.
/// Minimum sill relief (spill − bottom) for a basin to RESIST drainage integration — i.e. the
/// sill height an outlet would have to incise before geologic time breaches the basin. Basins
/// with less relief are shallow enough that integration is trivial, so they're breached. This
/// is a geologic-persistence / incision-resistance criterion and is DELIBERATELY SEPARATE from
/// `MIN_LAKE_DEPTH` (a rendering threshold) even though both are "depth": "deep enough to render
/// as a lake" and "deep enough to stay endorheic over geologic time" are different questions
/// (Codex round-2). Kept ≈ `MIN_INTEGRATION_SILL_RELIEF` is a PRAGMATIC proxy for the real
/// water-budget/incision model (see docs/specs/drainage-integration.md roadmap).
const MIN_INTEGRATION_SILL_RELIEF: f32 = 0.01;
/// Basins with footprint below this (steradians) are breached even if they clear the sill-relief
/// bar — a guard against deep erosion speckle surviving as a pin-prick lake. NOTE: this is a
/// crude SCALE-COUPLED heuristic (≈8100 km² Earth-equiv, NOT physically "tiny") and is currently
/// the dominant keep/breach lever among real basins; flagged for replacement with a
/// coherence/resolution-aware spike test (cell count, rim coherence). See docs roadmap.
const TINY_SPIKE_AREA: f32 = 0.0002;
/// Per-cell descent enforced when carving an outlet channel (elevation units). Small so the
/// breach is a thin notch, not a canyon.
const CARVE_SLOPE: f32 = 0.0002;

/// Minimum water body max depth for classification as a lake.
/// Shallower connected components are not considered lakes (filtered out).
/// This is checked per water body (connected component), not per basin.
pub const MIN_LAKE_DEPTH: f32 = 0.01;

impl Hydrology {
    /// Generate hydrology from elevation, crust, and precipitation.
    ///
    /// `precipitation` is the per-cell rainfall weight (mean ~1.0); it drives
    /// both flow accumulation and lake catchment budgets, so river density
    /// and lake levels respond to climate.
    pub fn generate(
        tessellation: &Tessellation,
        crust: &Crust,
        elevation: &Elevation,
        precipitation: &[f32],
        temperature: &[f32],
    ) -> Self {
        let continentality: Vec<f32> = (0..tessellation.num_cells())
            .map(|i| if crust.is_continental(i) { 1.0 } else { 0.0 })
            .collect();
        Self::generate_from_continentality(
            tessellation,
            &continentality,
            elevation,
            precipitation,
            temperature,
        )
    }

    pub fn generate_from_continentality(
        tessellation: &Tessellation,
        continentality: &[f32],
        elevation: &Elevation,
        precipitation: &[f32],
        temperature: &[f32],
    ) -> Self {
        let total = Instant::now();
        // Use raw elevation directly - MIN_LAKE_DEPTH filters spurious puddles
        let raw_elevation = &elevation.values;
        // Per-cell areas (steradians). Every water budget below is an AREA
        // integral, not a cell count — essential on the adaptive fine mesh where
        // land cells span ~50:1 in area and oceans are a few huge cells.
        let areas = tessellation.cell_areas();

        // Step 1: Identify ocean cells (connected below-sea-level touching oceanic crust)
        let t0 = Instant::now();
        let is_ocean = identify_ocean_cells_from_continentality(
            tessellation,
            continentality,
            raw_elevation,
            &areas,
        );
        log::info!("hydrology: identify ocean cells {:.2?}", t0.elapsed());

        // Drainage integration: carve outlet channels for basins geologic time would have
        // breached, so the present-climate hydrology drains them to the sea (not endorheic).
        let t0 = Instant::now();
        let (integrated_elevation, breached_cell) =
            integrate_basins(tessellation, raw_elevation, &areas, &is_ocean);
        let raw_elevation: &[f32] = &integrated_elevation;
        log::info!("hydrology: drainage integration {:.2?}", t0.elapsed());

        // Step 2: Priority-flood from ocean to detect all basins
        let t0 = Instant::now();
        let (filled_elevation, basin_id, mut basins, flood_parent) =
            priority_flood_with_basins(tessellation, raw_elevation, &areas, &is_ocean);
        log::info!(
            "hydrology: priority flood {:.2?} ({} basins)",
            t0.elapsed(),
            basins.len()
        );

        // Step 4: Compute drainage via steepest descent on filled surface
        // (needed before catchment calculation)
        let t0 = Instant::now();
        let drainage_dir =
            compute_steepest_descent(tessellation, &filled_elevation, &is_ocean, &flood_parent);
        log::info!("hydrology: drainage directions {:.2?}", t0.elapsed());

        // Step 5: Accumulate flow (precipitation-weighted)
        let t0 = Instant::now();
        let flow_accumulation = compute_flow_accumulations(&drainage_dir, precipitation, &areas);
        let global_mean_temperature = mean_temperature(temperature, &areas);
        log::info!("hydrology: flow accumulations {:.2?}", t0.elapsed());

        // Step 6: Compute each basin's LOCAL catchment budget (first-basin
        // attribution); inter-basin transfer is handled by the overflow cascade.
        let t0 = Instant::now();
        compute_basin_catchments(
            &mut basins,
            &basin_id,
            &drainage_dir,
            precipitation,
            temperature,
            &areas,
            global_mean_temperature,
        );
        log::info!("hydrology: basin catchments {:.2?}", t0.elapsed());

        // Step 7: Determine which basin each basin overflows into
        let t0 = Instant::now();
        compute_overflow_targets(&mut basins, &drainage_dir, &basin_id, &is_ocean);
        log::info!("hydrology: overflow targets {:.2?}", t0.elapsed());

        // Step 8: Calculate equilibrium water levels with overflow cascade
        let t0 = Instant::now();
        let climate_ratio = DEFAULT_CLIMATE_RATIO;
        calculate_water_levels(&mut basins, climate_ratio);
        log::info!("hydrology: water levels {:.2?}", t0.elapsed());

        // Over-connection audit (HEX3_OVERCONNECT_AUDIT): does a kept basin overflow only
        // because integration routed breached-pit runoff INTO it? Re-solve water levels with the
        // breached-pit precipitation masked out (≈ pre-integration, when that water was trapped
        // endorheically), on the SAME carved geometry, and count kept (sill-resisting) basins
        // that flip overflowing→not. A high fraction means the cascade manufactures connectivity
        // (Codex round-1 #5). Diagnostic only — does not affect output.
        if std::env::var("HEX3_OVERCONNECT_AUDIT").is_ok() {
            let masked_precip: Vec<f32> = precipitation
                .iter()
                .zip(&breached_cell)
                .map(|(&p, &b)| if b { 0.0 } else { p })
                .collect();
            let mut native = basins.clone();
            compute_basin_catchments(
                &mut native,
                &basin_id,
                &drainage_dir,
                &masked_precip,
                temperature,
                &areas,
                global_mean_temperature,
            );
            calculate_water_levels(&mut native, climate_ratio);
            let mut kept_overflowing = 0usize;
            let mut depends_on_routed = 0usize;
            for (b, nb) in basins.iter().zip(&native) {
                let resists =
                    (b.spill_elevation - b.bottom_elevation) >= MIN_INTEGRATION_SILL_RELIEF;
                if resists && b.is_overflowing() {
                    kept_overflowing += 1;
                    if !nb.is_overflowing() {
                        depends_on_routed += 1;
                    }
                }
            }
            let frac = if kept_overflowing > 0 {
                100.0 * depends_on_routed as f32 / kept_overflowing as f32
            } else {
                0.0
            };
            log::info!(
                "hydrology: OVERCONNECT AUDIT — {}/{} overflowing kept basins depend on routed-in breached-pit inflow ({:.1}%)",
                depends_on_routed,
                kept_overflowing,
                frac
            );
        }

        // Step 9: Extract connected water bodies from wet cells
        let t0 = Instant::now();
        let num_cells = tessellation.num_cells();
        let (water_bodies, cell_water_body) =
            extract_water_bodies(tessellation, raw_elevation, &basins, &basin_id, num_cells);
        log::info!(
            "hydrology: water bodies {:.2?} ({} bodies)",
            t0.elapsed(),
            water_bodies.len()
        );
        log::info!(
            "hydrology: total {:.2?} ({} cells)",
            total.elapsed(),
            tessellation.num_cells()
        );

        // Precip-weighted mean cell area, Σ(precip·area)/Σ(precip). Converts the
        // physical discharge in `flow_accumulation` back to a count-equivalent for
        // the legacy count-based thresholds (see field docs).
        let mean_cell_discharge = {
            let mut wsum = 0.0f64;
            let mut psum = 0.0f64;
            for i in 0..areas.len() {
                let p = precipitation[i].max(0.0);
                wsum += (p * areas[i]) as f64;
                psum += p as f64;
            }
            if psum > 0.0 {
                (wsum / psum) as f32
            } else {
                1.0
            }
        };

        Self {
            elevation: raw_elevation.to_vec(),
            filled_elevation,
            drainage_dir,
            flow_accumulation,
            is_ocean,
            basin_id,
            basins,
            water_bodies,
            cell_water_body,
            climate_ratio,
            mean_cell_discharge,
        }
    }

    /// Get the water state of a cell.
    ///
    /// Uses water body classification (connected components that pass depth
    /// threshold) rather than raw basin water level. This ensures consistency
    /// between what's rendered and what acts as a water sink.
    pub fn water_state(&self, cell_idx: usize) -> CellWaterState {
        if self.is_ocean[cell_idx] {
            return CellWaterState::Ocean;
        }

        // Check if cell is in a classified lake (water body that passes threshold)
        if let Some(wb_id) = self.cell_water_body[cell_idx] {
            if self.water_bodies[wb_id].is_lake {
                return CellWaterState::LakeWater;
            }
            // TODO: Could return Wetland here for shallow water bodies
        }

        // Cell is in a basin but not in a lake
        if self.basin_id[cell_idx].is_some() {
            CellWaterState::DryBasin
        } else {
            CellWaterState::Land
        }
    }

    /// Check if a cell is currently underwater (ocean or lake).
    pub fn is_submerged(&self, cell_idx: usize) -> bool {
        matches!(
            self.water_state(cell_idx),
            CellWaterState::Ocean | CellWaterState::LakeWater
        )
    }

    /// Check if a cell is ocean.
    pub fn is_ocean(&self, cell_idx: usize) -> bool {
        self.is_ocean[cell_idx]
    }

    /// Check if a cell is lake water (underwater, not ocean).
    pub fn is_lake_water(&self, cell_idx: usize) -> bool {
        self.water_state(cell_idx) == CellWaterState::LakeWater
    }

    /// Check if a cell is in a basin (wet or dry).
    pub fn is_in_basin(&self, cell_idx: usize) -> bool {
        self.basin_id[cell_idx].is_some()
    }

    /// Check if a cell is in a dry basin (above water level).
    pub fn is_dry_basin(&self, cell_idx: usize) -> bool {
        self.water_state(cell_idx) == CellWaterState::DryBasin
    }

    /// Get the basin for a cell (if any).
    pub fn basin(&self, cell_idx: usize) -> Option<&Basin> {
        self.basin_id[cell_idx].map(|id| &self.basins[id])
    }

    /// Get water depth at a cell (0 for non-water cells).
    pub fn water_depth(&self, cell_idx: usize) -> f32 {
        match self.water_state(cell_idx) {
            CellWaterState::Ocean => -self.elevation[cell_idx], // depth below sea level
            CellWaterState::LakeWater => {
                if let Some(basin_id) = self.basin_id[cell_idx] {
                    let basin = &self.basins[basin_id];
                    basin.water_level - self.elevation[cell_idx]
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// Get the downstream cell (where water flows to).
    pub fn downstream(&self, cell_idx: usize) -> Option<usize> {
        self.drainage_dir[cell_idx]
    }

    /// Count-equivalent flow at a cell: the physical discharge in
    /// `flow_accumulation` (precip × steradian) rescaled by `mean_cell_discharge`
    /// so it matches the old precip-weighted upstream cell count on an equal-area
    /// mesh. Use for count-based thresholds and log-scale river visualization;
    /// use the raw `flow_accumulation` where a physical discharge is wanted.
    pub fn flow_count_equiv(&self, cell_idx: usize) -> f32 {
        self.flow_accumulation[cell_idx] / self.mean_cell_discharge.max(1e-12)
    }

    /// Get river cells (land cells with flow above threshold). `threshold` is a
    /// count-equivalent (≈ upstream cells); it is converted to the physical
    /// discharge scale via `mean_cell_discharge` so it means the same on any mesh.
    pub fn river_cells(&self, threshold: f32) -> Vec<usize> {
        let flow_threshold = threshold * self.mean_cell_discharge;
        self.flow_accumulation
            .iter()
            .enumerate()
            .filter(|(i, &flow)| flow >= flow_threshold && !self.is_submerged(*i))
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute which cells are part of "major rivers".
    ///
    /// Algorithm: Start at major outlets (river mouths) and trace upstream,
    /// following all branches that exceed the branch threshold.
    ///
    /// - `outlet_threshold`: Minimum flow for a river mouth to be "major"
    /// - `branch_threshold`: Minimum flow to continue tracing a tributary
    ///
    /// At confluences, ALL upstream branches above branch_threshold are followed,
    /// naturally capturing major tributaries (like Ohio feeding into Mississippi).
    ///
    /// Returns a Vec<bool> indicating whether each cell is part of a major river.
    pub fn compute_major_river_cells(
        &self,
        outlet_threshold: f32,
        branch_threshold: f32,
    ) -> Vec<bool> {
        let n = self.drainage_dir.len();
        let mut is_major = vec![false; n];

        // Thresholds arrive as count-equivalents; scale to the physical discharge
        // units of `flow_accumulation` (see `mean_cell_discharge`).
        let outlet_threshold = outlet_threshold * self.mean_cell_discharge;
        let branch_threshold = branch_threshold * self.mean_cell_discharge;

        // Build reverse graph: for each cell, who drains INTO it
        let mut drains_into: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (cell, &downstream) in self.drainage_dir.iter().enumerate() {
            if let Some(d) = downstream {
                drains_into[d].push(cell);
            }
        }

        // Find major outlets: land cells that drain directly into water with high flow
        let mut queue: VecDeque<usize> = VecDeque::new();
        for cell in 0..n {
            if self.is_submerged(cell) {
                continue;
            }
            if self.flow_accumulation[cell] < outlet_threshold {
                continue;
            }
            // Check if this cell drains into water (ocean or lake)
            if let Some(downstream) = self.drainage_dir[cell] {
                if self.is_submerged(downstream) {
                    is_major[cell] = true;
                    queue.push_back(cell);
                }
            }
        }

        // BFS upstream: follow all branches above threshold
        while let Some(cell) = queue.pop_front() {
            for &upstream in &drains_into[cell] {
                if is_major[upstream] {
                    continue;
                }
                if self.is_submerged(upstream) {
                    continue;
                }
                // Follow this branch if it has enough flow
                if self.flow_accumulation[upstream] >= branch_threshold {
                    is_major[upstream] = true;
                    queue.push_back(upstream);
                }
            }
        }

        is_major
    }

    /// Update the climate ratio and recalculate all basin water levels.
    ///
    /// This recalculates water levels and re-extracts water bodies (connected
    /// components). Water body indices may change as lakes split/merge.
    pub fn set_climate_ratio(&mut self, tessellation: &Tessellation, new_ratio: f32) {
        self.climate_ratio = new_ratio.max(0.0);
        calculate_water_levels(&mut self.basins, self.climate_ratio);

        // Re-extract water bodies since water levels changed
        let num_cells = self.elevation.len();
        let (water_bodies, cell_water_body) = extract_water_bodies(
            tessellation,
            &self.elevation,
            &self.basins,
            &self.basin_id,
            num_cells,
        );
        self.water_bodies = water_bodies;
        self.cell_water_body = cell_water_body;
    }

    /// Get current climate ratio.
    pub fn climate_ratio(&self) -> f32 {
        self.climate_ratio
    }

    /// Get outflow paths from overflowing lakes.
    ///
    /// Returns a list of (basin_index, path) where path is the sequence of cells
    /// from the spill point to ocean or the next lake. Only includes basins
    /// that are currently overflowing.
    pub fn lake_outflow_paths(&self) -> Vec<(usize, Vec<usize>)> {
        let mut result = Vec::new();

        for (basin_idx, basin) in self.basins.iter().enumerate() {
            if !basin.is_overflowing() {
                continue;
            }

            // Trace from spill_target_cell downstream until we hit water
            let mut path = Vec::new();
            let mut cell = basin.spill_target_cell;

            loop {
                // Stop if we hit water (ocean or another lake)
                if self.is_submerged(cell) {
                    break;
                }

                path.push(cell);

                // Follow drainage
                match self.drainage_dir[cell] {
                    Some(next) => cell = next,
                    None => break, // No outlet
                }

                // Safety: prevent infinite loops
                if path.len() > 10000 {
                    break;
                }
            }

            if !path.is_empty() {
                result.push((basin_idx, path));
            }
        }

        result
    }
}

/// Identify ocean cells: connected below-sea-level regions that touch oceanic crust.
///
/// A connected component of elevation < 0 cells qualifies as ocean if:
/// - It contains at least one cell of oceanic crust
/// - It meets the minimum area threshold (fraction of total cells)
fn identify_ocean_cells_from_continentality(
    tessellation: &Tessellation,
    continentality: &[f32],
    elevation: &[f32],
    areas: &[f32],
) -> Vec<bool> {
    let n = tessellation.num_cells();
    // Minimum ocean size as a fraction of total surface AREA (not cell count):
    // on the adaptive mesh open ocean is a few huge cells, so a count threshold
    // would wrongly reject it.
    let total_area: f32 = areas.iter().sum();
    let min_ocean_area = total_area * MIN_OCEAN_AREA_FRACTION;
    let mut is_ocean = vec![false; n];
    let mut visited = vec![false; n];

    // Find connected components of below-sea-level cells
    for start in 0..n {
        if visited[start] || elevation[start] >= 0.0 {
            continue;
        }

        // BFS to find this connected component
        let mut component = Vec::new();
        let mut component_area = 0.0f32;
        let mut touches_oceanic = false;
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            component_area += areas[cell];

            // Check if this cell is on oceanic crust
            if continentality[cell] < 0.5 {
                touches_oceanic = true;
            }

            // Visit neighbors
            for &neighbor in tessellation.neighbors(cell) {
                if !visited[neighbor] && elevation[neighbor] < 0.0 {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        // Mark as ocean if it touches oceanic crust and meets minimum size
        if touches_oceanic && component_area >= min_ocean_area {
            for &cell in &component {
                is_ocean[cell] = true;
            }
        }
    }

    is_ocean
}

/// Priority-flood algorithm for depression filling with basin detection.
///
/// Starts from identified ocean cells and floods inward by elevation.
/// When entering a depression, immediately floods the entire basin.
///
/// Returns:
/// - filled elevation where every land cell can drain to ocean
/// - basin ID for each cell (None if not in a depression)
/// - vector of Basin structs (catchment_area and water_level not yet computed)
fn priority_flood_with_basins(
    tessellation: &Tessellation,
    elevation: &[f32],
    areas: &[f32],
    is_ocean_cell: &[bool],
) -> (Vec<f32>, Vec<Option<usize>>, Vec<Basin>, Vec<Option<usize>>) {
    let n = tessellation.num_cells();
    let mut filled = elevation.to_vec();
    let mut processed = vec![false; n];
    let mut basin_id: Vec<Option<usize>> = vec![None; n];
    let mut basins: Vec<Basin> = Vec::new();
    let mut flood_parent: Vec<Option<usize>> = vec![None; n];

    // Min-heap: process lowest elevations first
    let mut queue: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();

    // Seed with ocean cells (not just any below-sea-level cell)
    for cell in 0..n {
        if is_ocean_cell[cell] {
            queue.push(Reverse((OrderedFloat(elevation[cell]), cell)));
            processed[cell] = true;
        }
    }

    // Flood inward from ocean
    while let Some(Reverse((current_elev, cell))) = queue.pop() {
        let current_elev = current_elev.0;

        for &neighbor in tessellation.neighbors(cell) {
            if processed[neighbor] {
                continue;
            }

            if elevation[neighbor] >= current_elev {
                // Not a depression - normal processing
                processed[neighbor] = true;
                filled[neighbor] = elevation[neighbor];
                flood_parent[neighbor] = Some(cell);
                queue.push(Reverse((OrderedFloat(filled[neighbor]), neighbor)));
            } else {
                // Entering a depression - flood entire basin immediately
                let basin_idx = basins.len();
                let spill_elevation = current_elev;
                let mut basin_cells = Vec::new();
                let mut bottom_elevation = f32::INFINITY;

                // BFS to find and fill all connected cells below current_elev
                let mut basin_queue = VecDeque::new();
                basin_queue.push_back(neighbor);
                processed[neighbor] = true;
                filled[neighbor] = current_elev;
                flood_parent[neighbor] = Some(cell);
                basin_id[neighbor] = Some(basin_idx);
                basin_cells.push(neighbor);
                bottom_elevation = bottom_elevation.min(elevation[neighbor]);

                while let Some(basin_cell) = basin_queue.pop_front() {
                    for &basin_neighbor in tessellation.neighbors(basin_cell) {
                        if processed[basin_neighbor] {
                            continue;
                        }

                        if elevation[basin_neighbor] < current_elev {
                            // Still in the depression
                            processed[basin_neighbor] = true;
                            filled[basin_neighbor] = current_elev;
                            flood_parent[basin_neighbor] = Some(basin_cell);
                            basin_id[basin_neighbor] = Some(basin_idx);
                            basin_cells.push(basin_neighbor);
                            bottom_elevation = bottom_elevation.min(elevation[basin_neighbor]);
                            basin_queue.push_back(basin_neighbor);
                        } else {
                            // Far shore - add to main priority queue
                            processed[basin_neighbor] = true;
                            filled[basin_neighbor] = elevation[basin_neighbor];
                            flood_parent[basin_neighbor] = Some(basin_cell);
                            queue.push(Reverse((
                                OrderedFloat(filled[basin_neighbor]),
                                basin_neighbor,
                            )));
                        }
                    }
                }

                // Build sorted (elevation, area) for the hypsometric lookup; the
                // fill submerges cells by AREA, so the areas must track the
                // elevation sort order.
                let mut sorted: Vec<(f32, f32)> = basin_cells
                    .iter()
                    .map(|&c| (elevation[c], areas[c]))
                    .collect();
                sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                let sorted_elevations: Vec<f32> = sorted.iter().map(|&(e, _)| e).collect();
                let sorted_areas: Vec<f32> = sorted.iter().map(|&(_, a)| a).collect();
                let total_area: f32 = sorted_areas.iter().sum();

                // Create the basin (catchment, overflow_target, water_level computed later)
                basins.push(Basin {
                    cells: basin_cells,
                    sorted_elevations,
                    sorted_areas,
                    total_area,
                    spill_elevation,
                    bottom_elevation,
                    spill_target_cell: cell, // The cell outside basin where overflow exits
                    overflow_target: None,   // Computed after all basins identified
                    catchment_area: 0.0,     // Computed later
                    mean_temperature: 0.0,   // Computed later
                    evaporation_factor: 1.0, // Computed later
                    water_level: f32::NEG_INFINITY, // Computed later (dry by default)
                });
            }
        }
    }

    (filled, basin_id, basins, flood_parent)
}

/// Carve a monotonically-descending outlet channel from `start` (a basin's lowest cell) out
/// to the ocean, following the priority-flood `flood_parent` tree (which points oceanward).
/// Lowers only the barrier (the spill saddle + any rise above the running target); once the
/// path drops below the target it follows the natural descent. Edits `elevation` in place.
/// Carve a breached basin's outlet along the oceanward flood-parent path.
///
/// Basin-aware: if the path reaches a PRESERVED basin (`keep_basin[b]`), we stop there — the
/// breached pit now drains INTO that lake basin, which handles its own fill/overflow, rather
/// than the notch slicing straight through the lake to the sea (the collateral-carving bug:
/// 100% of cells cut through preserved basins were lake-capable). Edits `elevation` in place.
fn carve_outlet(
    elevation: &mut [f32],
    flood_parent: &[Option<usize>],
    is_ocean: &[bool],
    basin_id: &[Option<usize>],
    keep_basin: &[bool],
    start: usize,
    lowered: &mut Vec<usize>,
) {
    let n = elevation.len();
    let mut target = elevation[start];
    let mut c = start;
    let mut guard = 0;
    while !is_ocean[c] && guard < n {
        guard += 1;
        let Some(p) = flood_parent[c] else { break };
        // Reached a preserved (lake-capable) basin: route INTO it, don't carve through.
        if let Some(b) = basin_id[p] {
            if keep_basin[b] {
                break;
            }
        }
        target -= CARVE_SLOPE;
        if elevation[p] > target {
            elevation[p] = target; // carve the barrier down
            lowered.push(p); // record the cell we actually cut (for diagnostics)
        } else {
            target = elevation[p]; // already below target → follow the natural descent
        }
        c = p;
    }
}

/// Drainage integration: carve outlet channels for basins that geologic time would have
/// breached, so they drain to the sea instead of ponding endorheically. KEEPS lake-capable
/// basins (depth ≥ `MIN_LAKE_DEPTH`) so they pond + overflow as lakes-with-outlets; breaches
/// only basins too shallow to resist integration (`MIN_INTEGRATION_SILL_RELIEF`) plus genuinely
/// negligible deep spikes (`TINY_SPIKE_AREA`). The carve is basin-aware (routes into, not
/// through, preserved basins). Returns the carved elevation plus a per-cell mask of cells whose
/// pre-carve basin was BREACHED (the routed-in / "manufactured" runoff source — used by the
/// over-connection audit). Gated by HEX3_NO_DRAINAGE_INTEGRATION (A/B).
fn integrate_basins(
    tessellation: &Tessellation,
    elevation: &[f32],
    areas: &[f32],
    is_ocean: &[bool],
) -> (Vec<f32>, Vec<bool>) {
    let mut working = elevation.to_vec();
    let mut breached_cell = vec![false; elevation.len()];
    if std::env::var("HEX3_NO_DRAINAGE_INTEGRATION").is_ok() {
        return (working, breached_cell);
    }
    let (_filled, basin_id, basins, flood_parent) =
        priority_flood_with_basins(tessellation, elevation, areas, is_ocean);

    // KEEP a basin if it resists integration (sill relief ≥ MIN_INTEGRATION_SILL_RELIEF) AND is
    // not a negligible deep spike. Breach everything else. (Sill relief is the integration
    // criterion — distinct from MIN_LAKE_DEPTH, the lake-render threshold.)
    let resists: Vec<bool> = basins
        .iter()
        .map(|b| (b.spill_elevation - b.bottom_elevation) >= MIN_INTEGRATION_SILL_RELIEF)
        .collect();
    let keep: Vec<bool> = basins
        .iter()
        .zip(&resists)
        .map(|(b, &r)| r && b.total_area >= TINY_SPIKE_AREA)
        .collect();

    // Mark every cell whose basin will be breached (its runoff becomes routed-in inflow).
    for (cell, bid) in basin_id.iter().enumerate() {
        if let Some(b) = bid {
            if !keep[*b] {
                breached_cell[cell] = true;
            }
        }
    }

    let mut breached = 0;
    let mut lowered: Vec<usize> = Vec::new();
    for (bi, basin) in basins.iter().enumerate() {
        if keep[bi] {
            continue;
        }
        // The basin's lowest cell (start of the outlet channel).
        let bottom_cell = basin
            .cells
            .iter()
            .copied()
            .min_by(|&a, &b| elevation[a].total_cmp(&elevation[b]))
            .unwrap_or(basin.cells[0]);
        carve_outlet(
            &mut working,
            &flood_parent,
            is_ocean,
            &basin_id,
            &keep,
            bottom_cell,
            &mut lowered,
        );
        breached += 1;
    }

    // Collateral-carving diagnostic: how many cells we cut belong to a basin we
    // PRESERVED (kept). Should be ~0 now that carve_outlet stops at preserved basins.
    let collateral_preserved = lowered
        .iter()
        .filter(|&&c| basin_id[c].is_some_and(|b| keep[b]))
        .count();
    let n_kept = keep.iter().filter(|&&x| x).count();
    log::info!(
        "hydrology: drainage integration breached {}/{} basins | {} kept (sill≥{:.3} & area≥{:.4}) | carved {} cells, {} in preserved basins",
        breached,
        basins.len(),
        n_kept,
        MIN_INTEGRATION_SILL_RELIEF,
        TINY_SPIKE_AREA,
        lowered.len(),
        collateral_preserved,
    );
    (working, breached_cell)
}

/// For each cell, the basin that CAPTURES its runoff: the first basin reached by
/// following drainage (the cell's own basin if it is in one), or `None` if the
/// path reaches the ocean / a no-outlet cell without entering any basin.
///
/// This makes basins act as sinks: a cell's water is attributed to exactly one
/// basin (the first it reaches), never double-counted downstream. Inter-basin
/// transfer is handled separately by the overflow cascade in
/// `calculate_water_levels`. Memoized, O(n).
fn compute_capturing_basin(
    basin_id: &[Option<usize>],
    drainage_dir: &[Option<usize>],
) -> Vec<Option<usize>> {
    let n = basin_id.len();
    let mut capturing: Vec<Option<usize>> = vec![None; n];
    let mut resolved = vec![false; n];

    for start in 0..n {
        if resolved[start] {
            continue;
        }
        let mut path: Vec<usize> = Vec::new();
        let mut cur = start;
        let result;
        loop {
            if resolved[cur] {
                result = capturing[cur];
                break;
            }
            if let Some(b) = basin_id[cur] {
                result = Some(b);
                break;
            }
            path.push(cur);
            match drainage_dir[cur] {
                Some(next) => cur = next,
                None => {
                    result = None;
                    break;
                }
            }
            // Drainage is acyclic (steepest descent only targets strictly-lower
            // cells), but bound the walk defensively so a future change can't hang.
            if path.len() > n {
                result = None;
                break;
            }
        }
        for &c in &path {
            capturing[c] = result;
            resolved[c] = true;
        }
        if !resolved[cur] {
            capturing[cur] = result;
            resolved[cur] = true;
        }
    }
    capturing
}

/// Compute each basin's LOCAL catchment budget: the area-weighted discharge
/// (Σ precip·area) of the cells it captures (first basin on their drainage path),
/// plus the area-weighted mean temperature of those cells. Upstream basins that
/// capture their own water are NOT included here — if they overflow, that excess
/// is routed to this basin by the overflow cascade, so counting it here too would
/// double-count it.
fn compute_basin_catchments(
    basins: &mut [Basin],
    basin_id: &[Option<usize>],
    drainage_dir: &[Option<usize>],
    precipitation: &[f32],
    temperature: &[f32],
    areas: &[f32],
    global_mean_temperature: f32,
) {
    let capturing = compute_capturing_basin(basin_id, drainage_dir);

    let nb = basins.len();
    let mut discharge = vec![0.0f32; nb]; // Σ precip·area
    let mut area_sum = vec![0.0f32; nb]; // Σ area (mean-temperature divisor)
    let mut temperature_sum = vec![0.0f32; nb]; // Σ temp·area

    for cell in 0..capturing.len() {
        if let Some(b) = capturing[cell] {
            discharge[b] += precipitation[cell] * areas[cell];
            area_sum[b] += areas[cell];
            temperature_sum[b] += temperature[cell] * areas[cell];
        }
    }

    for (idx, basin) in basins.iter_mut().enumerate() {
        basin.catchment_area = discharge[idx];
        basin.mean_temperature = if area_sum[idx] > 0.0 {
            temperature_sum[idx] / area_sum[idx]
        } else {
            global_mean_temperature
        };
        basin.evaporation_factor =
            evaporation_factor_for_temperature(basin.mean_temperature, global_mean_temperature);
    }
}

/// Compute which basin each basin overflows into.
///
/// Follows drainage from each basin's spill_target_cell until hitting
/// either ocean (returns None) or another basin (returns Some(basin_id)).
fn compute_overflow_targets(
    basins: &mut [Basin],
    drainage_dir: &[Option<usize>],
    basin_id: &[Option<usize>],
    is_ocean: &[bool],
) {
    // Following drainage can in principle cycle (flat depressions filled to a
    // common level can leave `flood_parent`-based directions that loop without
    // reaching ocean or a basin tag). Bound the walk by the cell count — a
    // simple path can't visit more cells than exist — and treat an over-long
    // walk as "no outlet" rather than hanging worldgen.
    let max_steps = drainage_dir.len() + 1;
    for i in 0..basins.len() {
        let mut cell = basins[i].spill_target_cell;

        // Follow drainage until we hit ocean or another basin
        let mut steps = 0;
        loop {
            if is_ocean[cell] {
                basins[i].overflow_target = None;
                break;
            }
            if let Some(target_basin) = basin_id[cell] {
                // Don't overflow into ourselves (shouldn't happen, but safety check)
                basins[i].overflow_target = if target_basin != i {
                    Some(target_basin)
                } else {
                    None
                };
                break;
            }
            steps += 1;
            if steps > max_steps {
                log::warn!(
                    "hydrology: overflow walk for basin {i} exceeded {max_steps} steps \
                     (drainage cycle); treating as no outlet"
                );
                basins[i].overflow_target = None;
                break;
            }
            match drainage_dir[cell] {
                Some(next) => cell = next,
                None => {
                    // No outlet - treat as draining to void
                    basins[i].overflow_target = None;
                    break;
                }
            }
        }
    }
}

/// Assign a group id to each node of a functional graph (each node has at most
/// one successor). Nodes lying on a common cycle share a group id; every other
/// node becomes its own singleton group. Used to merge mutually-overflowing
/// basins (B spills into C and C spills into B) into a single lake.
fn group_overflow_cycles(succ: &[Option<usize>]) -> Vec<usize> {
    let n = succ.len();
    // 0 = unvisited, 1 = on current walk, 2 = done.
    let mut color = vec![0u8; n];
    let mut group_of = vec![usize::MAX; n];
    let mut next_group = 0usize;

    for start in 0..n {
        if color[start] != 0 {
            continue;
        }
        let mut stack: Vec<usize> = Vec::new();
        let mut cur = start;
        loop {
            match color[cur] {
                0 => {
                    color[cur] = 1;
                    stack.push(cur);
                    match succ[cur] {
                        Some(next) => cur = next,
                        None => break, // chain terminates (drains out): no cycle
                    }
                }
                1 => {
                    // Cycle: the suffix of `stack` from the first occurrence of cur.
                    let g = next_group;
                    next_group += 1;
                    let start_idx = stack.iter().position(|&x| x == cur).unwrap();
                    for &node in &stack[start_idx..] {
                        group_of[node] = g;
                    }
                    break;
                }
                _ => break, // merged into an already-processed chain: no new cycle
            }
        }
        for &node in &stack {
            color[node] = 2;
        }
    }

    // Everything not on a cycle is its own singleton group.
    for g in group_of.iter_mut() {
        if *g == usize::MAX {
            *g = next_group;
            next_group += 1;
        }
    }
    group_of
}

/// Calculate equilibrium water levels with an overflow cascade.
///
/// At equilibrium inflow = evaporation, so a lake's surface area =
/// catchment_discharge × (precip/evap) = `catchment_area × effective_ratio`.
///
/// Basins that mutually overflow are MERGED into one lake group: they equilibrate
/// to a shared surface bounded by the group's lowest external saddle, with a
/// combined catchment and hypsometry. Groups are then solved in TOPOLOGICAL order
/// of the (now acyclic) overflow graph, so a group's received overflow is known
/// before its level is computed — fixing the double-count (catchment is now the
/// local, first-basin discharge) and the old spill-elevation cascade ordering.
fn calculate_water_levels(basins: &mut [Basin], climate_ratio: f32) {
    if basins.is_empty() {
        return;
    }
    let n = basins.len();

    // 1. Merge mutually-overflowing basins. The overflow graph is functional
    //    (each basin has ≤1 target), so its only multi-node SCCs are simple cycles.
    let succ: Vec<Option<usize>> = basins.iter().map(|b| b.overflow_target).collect();
    let group_of = group_overflow_cycles(&succ);
    let num_groups = group_of.iter().copied().max().map_or(0, |m| m + 1);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); num_groups];
    for (b, &g) in group_of.iter().enumerate() {
        members[g].push(b);
    }

    // 2. Per-group properties: combined catchment, evaporation, the external exit
    //    (lowest member spill leaving the group), and the fillable hypsometry
    //    (cells below that exit — cells above it can't hold water).
    struct Group {
        catchment: f32,
        sorted_elev: Vec<f32>,
        sorted_area: Vec<f32>,
        capacity: f32,
        spill_elevation: f32,
        bottom_elevation: f32,
        overflow_group: Option<usize>,
        eff_ratio: f32,
    }

    let mut groups: Vec<Group> = Vec::with_capacity(num_groups);
    for mem in &members {
        // External spill: lowest member spill whose overflow leaves the group.
        let mut spill_elevation = f32::INFINITY;
        let mut overflow_group: Option<usize> = None;
        for &b in mem {
            let external = match basins[b].overflow_target {
                Some(t) => group_of[t] != group_of[b], // points to a different group
                None => true,                          // ocean / void: an external exit
            };
            if external && basins[b].spill_elevation < spill_elevation {
                spill_elevation = basins[b].spill_elevation;
                overflow_group = basins[b].overflow_target.map(|t| group_of[t]);
            }
        }
        if !spill_elevation.is_finite() {
            // A multi-member cycle: every member's overflow_target points at
            // another member, so none is "external" and we reach here. The merged
            // lake's TRUE external outlet is the lowest saddle out of the cell
            // union, which per-basin spill data can't recover (each member's spill
            // is an INTERNAL connector to another member). We approximate the
            // merged lake as endorheic, and cap at the HIGHEST member rim so the
            // `*e < spill_elevation` filter keeps every union cell fillable (a min
            // cap would wrongly drop the higher member's cells). Cycles are rare;
            // a rigorous external outlet would need saddle tracking in the
            // priority-flood. See docs/algorithm-audit-2026-06-15.md.
            spill_elevation = mem
                .iter()
                .map(|&b| basins[b].spill_elevation)
                .fold(f32::NEG_INFINITY, f32::max);
            overflow_group = None;
        }

        let mut pairs: Vec<(f32, f32)> = Vec::new();
        let mut bottom_elevation = f32::INFINITY;
        let mut catchment = 0.0f32;
        let mut evap_accum = 0.0f32;
        let mut evap_weight = 0.0f32;
        for &b in mem {
            catchment += basins[b].catchment_area;
            let w = basins[b].catchment_area.max(1e-12);
            evap_accum += basins[b].evaporation_factor * w;
            evap_weight += w;
            bottom_elevation = bottom_elevation.min(basins[b].bottom_elevation);
            for (e, a) in basins[b]
                .sorted_elevations
                .iter()
                .zip(basins[b].sorted_areas.iter())
            {
                if *e < spill_elevation {
                    pairs.push((*e, *a));
                }
            }
        }
        pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        let capacity: f32 = pairs.iter().map(|&(_, a)| a).sum();
        let sorted_elev: Vec<f32> = pairs.iter().map(|&(e, _)| e).collect();
        let sorted_area: Vec<f32> = pairs.iter().map(|&(_, a)| a).collect();
        let evap_factor = if evap_weight > 0.0 {
            evap_accum / evap_weight
        } else {
            1.0
        };
        let eff_ratio = if evap_factor > 0.0 {
            climate_ratio / evap_factor
        } else {
            climate_ratio
        };

        groups.push(Group {
            catchment,
            sorted_elev,
            sorted_area,
            capacity,
            spill_elevation,
            bottom_elevation,
            overflow_group,
            eff_ratio,
        });
    }

    // 3. Topological order of the group overflow DAG (Kahn). Sources (most
    //    upstream) first, so received overflow is known before a group is solved.
    let mut indeg = vec![0usize; num_groups];
    for grp in &groups {
        if let Some(t) = grp.overflow_group {
            indeg[t] += 1;
        }
    }
    let mut queue: Vec<usize> = (0..num_groups).filter(|&g| indeg[g] == 0).collect();
    let mut topo: Vec<usize> = Vec::with_capacity(num_groups);
    while let Some(g) = queue.pop() {
        topo.push(g);
        if let Some(t) = groups[g].overflow_group {
            indeg[t] -= 1;
            if indeg[t] == 0 {
                queue.push(t);
            }
        }
    }

    // 4. Solve each group's level in topological order, cascading overflow.
    let mut received = vec![0.0f32; num_groups];
    let mut group_level = vec![f32::NEG_INFINITY; num_groups];
    for &g in &topo {
        let effective_catchment = groups[g].catchment + received[g];
        let eff_ratio = groups[g].eff_ratio;
        let target_area = effective_catchment * eff_ratio;
        let capacity = groups[g].capacity;
        let spill = groups[g].spill_elevation;
        let bottom = groups[g].bottom_elevation;

        // A group too shallow overall doesn't form a lake even if it overflows.
        if spill - bottom < MIN_LAKE_DEPTH {
            group_level[g] = bottom - 1.0;
            continue;
        }

        group_level[g] =
            if target_area <= 0.0 || eff_ratio <= 0.0 || groups[g].sorted_elev.is_empty() {
                bottom - 1.0
            } else if target_area >= capacity {
                // Overflows to the external saddle; route excess to the downstream group.
                let overflow_amount = effective_catchment - (capacity / eff_ratio);
                if let Some(t) = groups[g].overflow_group {
                    received[t] += overflow_amount.max(0.0);
                }
                spill
            } else {
                // Partial fill by AREA: submerge whole cells whose cumulative area
                // fits; the level sits between the last submerged cell and the next.
                let mut cumulative = 0.0f32;
                let mut last_submerged: Option<usize> = None;
                for (idx, &area) in groups[g].sorted_area.iter().enumerate() {
                    if cumulative + area > target_area {
                        break;
                    }
                    cumulative += area;
                    last_submerged = Some(idx);
                }
                match last_submerged {
                    None => bottom - 1.0,
                    Some(k) if k + 1 < groups[g].sorted_elev.len() => {
                        (groups[g].sorted_elev[k] + groups[g].sorted_elev[k + 1]) / 2.0
                    }
                    Some(_) => spill,
                }
            };
    }

    // 5. Write each group's surface level back to its member basins.
    for b in 0..n {
        basins[b].water_level = group_level[group_of[b]];
    }
}

/// Compute drainage direction via steepest descent on filled elevation.
///
/// Each cell drains to the neighbor of greatest downhill SLOPE — the elevation
/// drop normalized by inter-cell (chord) distance — not merely the lowest
/// neighbor. On the adaptive fine mesh cell spacing varies ~50:1, so picking the
/// lowest-elevation neighbor would bias drainage toward distant large cells over
/// nearer steeper ones; normalizing by distance gives the actual gradient.
/// Only strictly-lower neighbors are considered, so the result is acyclic.
fn compute_steepest_descent(
    tessellation: &Tessellation,
    filled_elevation: &[f32],
    is_ocean: &[bool],
    flood_parent: &[Option<usize>],
) -> Vec<Option<usize>> {
    let n = tessellation.num_cells();
    let mut drainage: Vec<Option<usize>> = vec![None; n];

    for cell in 0..n {
        // Ocean cells don't drain anywhere (they're the sink)
        if is_ocean[cell] {
            continue;
        }

        let cell_elev = filled_elevation[cell];
        let cell_pos = tessellation.cell_center(cell);

        // Find the strictly-lower neighbor of greatest downhill slope.
        let mut best_neighbor: Option<usize> = None;
        let mut best_slope = 0.0f32;

        for &neighbor in tessellation.neighbors(cell) {
            let drop = cell_elev - filled_elevation[neighbor];
            if drop <= 0.0 {
                continue;
            }
            let dist = (cell_pos - tessellation.cell_center(neighbor))
                .length()
                .max(1e-12);
            let slope = drop / dist;
            if slope > best_slope {
                best_slope = slope;
                best_neighbor = Some(neighbor);
            }
        }

        // On flats (no strictly-lower neighbor), fall back to the priority-flood parent.
        // flood_parent always points to an equal-or-lower filled cell (it's the BFS
        // tree rooted at ocean), so this keeps drainage acyclic and ocean-reaching.
        if best_neighbor.is_none() {
            best_neighbor = flood_parent[cell];
        }

        drainage[cell] = best_neighbor;
    }

    drainage
}

/// Area-weighted discharge accumulated down the drainage graph:
/// `flow[c] = Σ precip·area` over every cell whose drainage path passes through
/// `c` (including `c`). Used for river extraction and the fine-mesh density
/// prior. NOTE: this routes through filled depressions (basins are not sinks
/// here), so it is NOT the basin water budget — that is computed by
/// `compute_basin_catchments`, which attributes each cell to the first basin its
/// drainage reaches.
fn compute_flow_accumulations(
    drainage_dir: &[Option<usize>],
    precipitation: &[f32],
    areas: &[f32],
) -> Vec<f32> {
    let n = drainage_dir.len();

    // Count how many cells drain into each cell (upstream count)
    let mut upstream_count = vec![0usize; n];
    for downstream in drainage_dir.iter().flatten() {
        upstream_count[*downstream] += 1;
    }

    let mut precipitation_flow: Vec<f32> = (0..n).map(|i| precipitation[i] * areas[i]).collect();
    let mut remaining_upstream = upstream_count.clone();
    let mut ready: Vec<usize> = (0..n).filter(|&c| upstream_count[c] == 0).collect();

    while let Some(cell) = ready.pop() {
        if let Some(downstream) = drainage_dir[cell] {
            precipitation_flow[downstream] += precipitation_flow[cell];
            remaining_upstream[downstream] -= 1;
            if remaining_upstream[downstream] == 0 {
                ready.push(downstream);
            }
        }
    }

    precipitation_flow
}

/// Area-weighted global mean surface temperature (the neutral point of the
/// per-basin evaporation factor). Area-weighted so the adaptive mesh's many
/// tiny mountain cells don't bias the mean cold.
fn mean_temperature(temperature: &[f32], areas: &[f32]) -> f32 {
    let mut wsum = 0.0f64;
    let mut asum = 0.0f64;
    for i in 0..temperature.len() {
        wsum += (temperature[i] * areas[i]) as f64;
        asum += areas[i] as f64;
    }
    if asum > 0.0 {
        (wsum / asum) as f32
    } else {
        0.0
    }
}

fn evaporation_factor_for_temperature(mean_temp: f32, global_mean_temp: f32) -> f32 {
    ((mean_temp - global_mean_temp) * EVAP_TEMP_SENSITIVITY).exp()
}

/// Extract connected water bodies from wet cells in basins.
///
/// For each basin with water, finds connected components of underwater cells.
/// Each component becomes a WaterBody, classified as lake if max_depth >= threshold.
///
/// Returns (water_bodies, cell_water_body) where cell_water_body[i] is the
/// water body index for cell i (or None if not in a water body).
fn extract_water_bodies(
    tessellation: &Tessellation,
    elevation: &[f32],
    basins: &[Basin],
    basin_id: &[Option<usize>],
    num_cells: usize,
) -> (Vec<WaterBody>, Vec<Option<usize>>) {
    let mut water_bodies = Vec::new();
    let mut cell_water_body: Vec<Option<usize>> = vec![None; num_cells];
    let mut visited = vec![false; num_cells];

    // Process each basin
    for (bid, basin) in basins.iter().enumerate() {
        // Skip dry basins
        if basin.water_level <= basin.bottom_elevation {
            continue;
        }

        // Find connected components of underwater cells within this basin
        for &start_cell in &basin.cells {
            // Skip if already visited or not underwater
            if visited[start_cell] || elevation[start_cell] >= basin.water_level {
                continue;
            }

            // BFS to find this connected component
            let mut component_cells = Vec::new();
            let mut min_elev = f32::INFINITY;
            let mut queue = VecDeque::new();

            queue.push_back(start_cell);
            visited[start_cell] = true;

            while let Some(cell) = queue.pop_front() {
                component_cells.push(cell);
                min_elev = min_elev.min(elevation[cell]);

                // Visit underwater neighbors in the same basin
                for &neighbor in tessellation.neighbors(cell) {
                    if visited[neighbor] {
                        continue;
                    }
                    // Must be in same basin and underwater
                    if basin_id[neighbor] != Some(bid) {
                        continue;
                    }
                    if elevation[neighbor] >= basin.water_level {
                        continue;
                    }

                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }

            // Create water body from this component
            let max_depth = basin.water_level - min_elev;
            let is_lake = max_depth >= MIN_LAKE_DEPTH;

            let wb_id = water_bodies.len();

            // Only assign cell_water_body for cells in lakes (not shallow puddles)
            if is_lake {
                for &cell in &component_cells {
                    cell_water_body[cell] = Some(wb_id);
                }
            }

            water_bodies.push(WaterBody {
                cells: component_cells,
                basin_id: bid,
                max_depth,
                is_lake,
            });
        }
    }

    (water_bodies, cell_water_body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_basin(evaporation_factor: f32) -> Basin {
        Basin {
            cells: (0..10).collect(),
            sorted_elevations: (0..10).map(|i| i as f32).collect(),
            sorted_areas: vec![1.0; 10],
            total_area: 10.0,
            spill_elevation: 10.0,
            bottom_elevation: 0.0,
            spill_target_cell: 0,
            overflow_target: None,
            catchment_area: 20.0,
            mean_temperature: 0.5,
            evaporation_factor,
            water_level: f32::NEG_INFINITY,
        }
    }

    #[test]
    fn evaporation_factor_is_neutral_at_global_mean() {
        assert_eq!(evaporation_factor_for_temperature(0.5, 0.5), 1.0);
        assert!(evaporation_factor_for_temperature(0.7, 0.5) > 1.0);
        assert!(evaporation_factor_for_temperature(0.3, 0.5) < 1.0);
    }

    #[test]
    fn mean_temperature_basin_reproduces_global_ratio() {
        let mut basins = vec![test_basin(1.0)];
        calculate_water_levels(&mut basins, 0.2);
        assert_eq!(basins[0].water_level, 3.5);
    }

    /// Each cell is attributed to the FIRST basin its drainage reaches, so an
    /// upstream basin's water is NOT also counted in a downstream basin (the old
    /// filled-graph accounting double-counted it). Here cells 0,1,2 drain into
    /// basin 0; cells 3,4 drain into basin 1 — basin 1 does NOT inherit basin 0's
    /// catchment. (Inter-basin transfer, if basin 0 overflowed, would be the
    /// cascade's job, not the catchment's.)
    #[test]
    fn basin_catchments_use_local_first_basin_attribution() {
        let mut basins = vec![
            Basin {
                cells: vec![2],
                sorted_elevations: vec![0.0],
                sorted_areas: vec![1.0],
                total_area: 1.0,
                spill_elevation: 1.0,
                bottom_elevation: 0.0,
                spill_target_cell: 1,
                overflow_target: None,
                catchment_area: 0.0,
                mean_temperature: 0.0,
                evaporation_factor: 1.0,
                water_level: f32::NEG_INFINITY,
            },
            Basin {
                cells: vec![4],
                sorted_elevations: vec![0.0],
                sorted_areas: vec![1.0],
                total_area: 1.0,
                spill_elevation: 1.0,
                bottom_elevation: 0.0,
                spill_target_cell: 3,
                overflow_target: None,
                catchment_area: 0.0,
                mean_temperature: 0.0,
                evaporation_factor: 1.0,
                water_level: f32::NEG_INFINITY,
            },
        ];
        let basin_id = vec![None, None, Some(0), None, Some(1)];
        let drainage_dir = vec![Some(1), Some(2), Some(3), Some(4), None];
        let precipitation = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let temperature = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let areas = vec![1.0; precipitation.len()];

        compute_basin_catchments(
            &mut basins,
            &basin_id,
            &drainage_dir,
            &precipitation,
            &temperature,
            &areas,
            30.0,
        );

        // basin 0 captures cells {0,1,2}: precip 1+2+3=6, temp (10+20+30)/3=20.
        assert_eq!(basins[0].catchment_area, 6.0);
        assert_eq!(basins[0].mean_temperature, 20.0);
        // basin 1 captures only {3,4}: precip 4+5=9, temp (40+50)/2=45 — it does
        // NOT include basin 0's cells (old buggy value was 15 / 30).
        assert_eq!(basins[1].catchment_area, 9.0);
        assert_eq!(basins[1].mean_temperature, 45.0);
    }

    /// The hypsometric fill measures the submerged surface by AREA, not cell
    /// count: a basin with unequal cell areas fills to the level where the
    /// cumulative cell area reaches the target, not where the cell *count* does.
    #[test]
    fn lake_level_fills_by_area_not_cell_count() {
        let mut basins = vec![Basin {
            cells: (0..4).collect(),
            sorted_elevations: vec![0.0, 1.0, 2.0, 3.0],
            sorted_areas: vec![1.0, 2.0, 3.0, 4.0],
            total_area: 10.0,
            spill_elevation: 4.0,
            bottom_elevation: 0.0,
            spill_target_cell: 0,
            overflow_target: None,
            catchment_area: 10.0,
            mean_temperature: 0.5,
            evaporation_factor: 1.0,
            water_level: f32::NEG_INFINITY,
        }];
        // target_area = 10 * 0.5 = 5.0 sr. Cumulative area: cell0=1, +cell1=3,
        // +cell2=6 (> 5) -> submerge cells 0,1; level sits between elev 1 and 2.
        // A cell-COUNT fill (floor(5)=5 cells >= 4) would instead overflow to 4.0.
        calculate_water_levels(&mut basins, 0.5);
        assert_eq!(basins[0].water_level, 1.5);
    }

    /// Overflow cascades in TOPOLOGICAL order, not by spill elevation. An upstream
    /// basin with a LOWER spill overflows into a downstream basin with a HIGHER
    /// spill; the downstream basin must still receive that water. The old
    /// spill-elevation-descending single pass processed the downstream basin
    /// first and silently dropped the upstream overflow.
    #[test]
    fn overflow_cascade_follows_topology_not_spill_elevation() {
        let mut basins = vec![
            // Upstream basin A: low spill, tiny capacity, big catchment -> overflows.
            Basin {
                cells: vec![0],
                sorted_elevations: vec![0.0],
                sorted_areas: vec![0.5],
                total_area: 0.5,
                spill_elevation: 1.0, // LOWER than B's spill
                bottom_elevation: 0.0,
                spill_target_cell: 0,
                overflow_target: Some(1),
                catchment_area: 10.0,
                mean_temperature: 0.5,
                evaporation_factor: 1.0,
                water_level: f32::NEG_INFINITY,
            },
            // Downstream basin B: high spill, no catchment of its own.
            Basin {
                cells: vec![1, 2],
                sorted_elevations: vec![2.0, 3.0],
                sorted_areas: vec![1.0, 1.0],
                total_area: 2.0,
                spill_elevation: 5.0, // HIGHER than A's spill
                bottom_elevation: 2.0,
                spill_target_cell: 1,
                overflow_target: None,
                catchment_area: 0.0,
                mean_temperature: 0.5,
                evaporation_factor: 1.0,
                water_level: f32::NEG_INFINITY,
            },
        ];
        calculate_water_levels(&mut basins, 1.0);
        // A overflows (target 10 >> capacity 0.5).
        assert!(basins[0].is_overflowing());
        // B received A's overflow despite its higher spill -> it holds water.
        assert!(
            basins[1].has_water(),
            "downstream basin must receive upstream overflow: level={}",
            basins[1].water_level
        );
    }

    /// Two basins that overflow into each other are merged into one lake group
    /// with a shared surface level and a combined catchment + hypsometry.
    #[test]
    fn mutually_overflowing_basins_merge_to_one_level() {
        let mut basins = vec![
            Basin {
                cells: vec![0, 1],
                sorted_elevations: vec![0.0, 2.0],
                sorted_areas: vec![1.0, 1.0],
                total_area: 2.0,
                spill_elevation: 4.0,
                bottom_elevation: 0.0,
                spill_target_cell: 0,
                overflow_target: Some(1),
                catchment_area: 1.5,
                mean_temperature: 0.5,
                evaporation_factor: 1.0,
                water_level: f32::NEG_INFINITY,
            },
            Basin {
                cells: vec![2, 3],
                sorted_elevations: vec![1.0, 3.0],
                sorted_areas: vec![1.0, 1.0],
                total_area: 2.0,
                spill_elevation: 4.0,
                bottom_elevation: 1.0,
                spill_target_cell: 2,
                overflow_target: Some(0),
                catchment_area: 1.5,
                mean_temperature: 0.5,
                evaporation_factor: 1.0,
                water_level: f32::NEG_INFINITY,
            },
        ];
        calculate_water_levels(&mut basins, 1.0);
        // Shared surface: combined catchment 3.0, ratio 1.0 -> target area 3.0 over
        // the merged hypsometry [(0,1),(1,1),(2,1),(3,1)] submerges 3 cells; level
        // sits between elev 2 and 3 = 2.5. Both members read the same level.
        assert_eq!(basins[0].water_level, basins[1].water_level);
        assert_eq!(basins[0].water_level, 2.5);
    }

    /// A merged cycle with UNEQUAL member spills must keep the higher member's
    /// cells fillable (cap at the max rim, not the min). With the old min cap the
    /// higher basin's cells were filtered out and the lake capped at the low rim.
    #[test]
    fn merged_cycle_with_unequal_spills_keeps_higher_cells() {
        let mut basins = vec![
            Basin {
                cells: vec![0, 1],
                sorted_elevations: vec![0.0, 1.0],
                sorted_areas: vec![1.0, 1.0],
                total_area: 2.0,
                spill_elevation: 2.0, // low rim
                bottom_elevation: 0.0,
                spill_target_cell: 0,
                overflow_target: Some(1),
                catchment_area: 1.5,
                mean_temperature: 0.5,
                evaporation_factor: 1.0,
                water_level: f32::NEG_INFINITY,
            },
            Basin {
                cells: vec![2, 3],
                sorted_elevations: vec![3.0, 5.0],
                sorted_areas: vec![1.0, 1.0],
                total_area: 2.0,
                spill_elevation: 6.0, // high rim
                bottom_elevation: 3.0,
                spill_target_cell: 2,
                overflow_target: Some(0),
                catchment_area: 1.5,
                mean_temperature: 0.5,
                evaporation_factor: 1.0,
                water_level: f32::NEG_INFINITY,
            },
        ];
        calculate_water_levels(&mut basins, 1.0);
        // Merged hypsometry [(0,1),(1,1),(3,1),(5,1)], target 3.0 submerges 3 cells
        // -> level between elev 3 and 5 = 4.0, well above the low rim (2.0) the old
        // min cap would have produced. Both members share it.
        assert_eq!(basins[0].water_level, basins[1].water_level);
        assert_eq!(basins[0].water_level, 4.0);
    }
}
