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

use ordered_float::OrderedFloat;

use super::{Dynamics, Elevation, PlateType, Plates, Tessellation};

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
    /// Spill elevation - maximum possible water level.
    pub spill_elevation: f32,
    /// Bottom elevation - lowest point in the basin.
    pub bottom_elevation: f32,
    /// The cell just outside the basin where overflow exits.
    /// Following drainage from here leads to ocean or another basin.
    pub spill_target_cell: usize,
    /// Which basin this one overflows into (None = drains to ocean).
    pub overflow_target: Option<usize>,
    /// Number of upstream cells that drain into this basin.
    pub catchment_area: usize,
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
}

/// Minimum ocean component size as fraction of total cells.
const MIN_OCEAN_AREA_FRACTION: f32 = 0.001;

/// Default climate ratio (precipitation / evaporation).
/// Values > 1.0 = wet climate (more lakes)
/// Values < 1.0 = arid climate (fewer lakes)
/// At equilibrium: lake_surface_area = catchment_area × climate_ratio
pub const DEFAULT_CLIMATE_RATIO: f32 = 0.15;

/// Minimum water body max depth for classification as a lake.
/// Shallower connected components are not considered lakes (filtered out).
/// This is checked per water body (connected component), not per basin.
pub const MIN_LAKE_DEPTH: f32 = 0.01;

impl Hydrology {
    /// Generate hydrology from elevation and plate data.
    pub fn generate(
        tessellation: &Tessellation,
        plates: &Plates,
        dynamics: &Dynamics,
        elevation: &Elevation,
    ) -> Self {
        // Use raw elevation directly - MIN_LAKE_DEPTH filters spurious puddles
        let raw_elevation = &elevation.values;

        // Step 1: Identify ocean cells (connected below-sea-level touching oceanic crust)
        let is_ocean = identify_ocean_cells(tessellation, plates, dynamics, raw_elevation);

        // Step 2: Priority-flood from ocean to detect all basins
        let (filled_elevation, basin_id, mut basins, flood_parent) =
            priority_flood_with_basins(tessellation, raw_elevation, &is_ocean);

        // Step 4: Compute drainage via steepest descent on filled surface
        // (needed before catchment calculation)
        let drainage_dir =
            compute_steepest_descent(tessellation, &filled_elevation, &is_ocean, &flood_parent);

        // Step 5: Accumulate flow
        let flow_accumulation = compute_flow_accumulation(tessellation, &drainage_dir);

        // Step 6: Compute catchment area for each basin
        compute_basin_catchments(&mut basins, &basin_id, &drainage_dir, &flow_accumulation);

        // Step 7: Determine which basin each basin overflows into
        compute_overflow_targets(&mut basins, &drainage_dir, &basin_id, &is_ocean);

        // Step 8: Calculate equilibrium water levels with overflow cascade
        let climate_ratio = DEFAULT_CLIMATE_RATIO;
        calculate_water_levels(&mut basins, climate_ratio);

        // Step 9: Extract connected water bodies from wet cells
        let num_cells = tessellation.num_cells();
        let (water_bodies, cell_water_body) =
            extract_water_bodies(tessellation, raw_elevation, &basins, &basin_id, num_cells);

        Self {
            elevation: raw_elevation.clone(),
            filled_elevation,
            drainage_dir,
            flow_accumulation,
            is_ocean,
            basin_id,
            basins,
            water_bodies,
            cell_water_body,
            climate_ratio,
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

    /// Get river cells (land cells with flow above threshold).
    pub fn river_cells(&self, threshold: f32) -> Vec<usize> {
        self.flow_accumulation
            .iter()
            .enumerate()
            .filter(|(i, &flow)| flow >= threshold && !self.is_submerged(*i))
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
/// - It contains at least one cell on oceanic plate crust
/// - It meets the minimum area threshold (fraction of total cells)
fn identify_ocean_cells(
    tessellation: &Tessellation,
    plates: &Plates,
    dynamics: &Dynamics,
    elevation: &[f32],
) -> Vec<bool> {
    let n = tessellation.num_cells();
    let min_ocean_area = ((n as f32) * MIN_OCEAN_AREA_FRACTION).ceil() as usize;
    let mut is_ocean = vec![false; n];
    let mut visited = vec![false; n];

    // Find connected components of below-sea-level cells
    for start in 0..n {
        if visited[start] || elevation[start] >= 0.0 {
            continue;
        }

        // BFS to find this connected component
        let mut component = Vec::new();
        let mut touches_oceanic = false;
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited[start] = true;

        while let Some(cell) = queue.pop_front() {
            component.push(cell);

            // Check if this cell is on oceanic crust
            let plate_id = plates.cell_plate[cell] as usize;
            if dynamics.plate_type(plate_id) == PlateType::Oceanic {
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
        if touches_oceanic && component.len() >= min_ocean_area {
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

                // Build sorted elevations for hypsometric lookup
                let mut sorted_elevations: Vec<f32> =
                    basin_cells.iter().map(|&c| elevation[c]).collect();
                sorted_elevations.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Create the basin (catchment, overflow_target, water_level computed later)
                basins.push(Basin {
                    cells: basin_cells,
                    sorted_elevations,
                    spill_elevation,
                    bottom_elevation,
                    spill_target_cell: cell, // The cell outside basin where overflow exits
                    overflow_target: None,   // Computed after all basins identified
                    catchment_area: 0,       // Computed later
                    water_level: f32::NEG_INFINITY, // Computed later (dry by default)
                });
            }
        }
    }

    (filled, basin_id, basins, flood_parent)
}

/// Compute catchment area for each basin.
///
/// The catchment is the total upstream area that drains into the basin.
/// We use flow accumulation at cells that drain into basin cells.
fn compute_basin_catchments(
    basins: &mut [Basin],
    basin_id: &[Option<usize>],
    drainage_dir: &[Option<usize>],
    flow_accumulation: &[f32],
) {
    // For each basin, find cells that drain INTO it (but aren't in it)
    // The catchment is the sum of flow at the entry points

    for (idx, basin) in basins.iter_mut().enumerate() {
        // Start with basin area (each cell collects rainfall)
        let mut catchment = basin.cells.len();

        // Also count cells that drain into the basin from outside
        for (cell, &downstream) in drainage_dir.iter().enumerate() {
            if let Some(downstream) = downstream {
                // If this cell drains into a basin cell, and isn't itself in the basin
                if basin_id[downstream] == Some(idx) && basin_id[cell] != Some(idx) {
                    // This cell's flow feeds into the basin
                    catchment += flow_accumulation[cell] as usize;
                }
            }
        }

        basin.catchment_area = catchment;
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
    for i in 0..basins.len() {
        let mut cell = basins[i].spill_target_cell;

        // Follow drainage until we hit ocean or another basin
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

/// Calculate equilibrium water levels for each basin using evaporation balance,
/// with overflow cascade from higher basins to lower ones.
///
/// At equilibrium: inflow = evaporation
///   inflow ∝ catchment_area × precipitation
///   evaporation ∝ lake_surface_area × evaporation_rate
///
/// Therefore: lake_surface_area = catchment_area × (precip / evap) = catchment × climate_ratio
///
/// When a basin overflows, excess water cascades to downstream basins.
/// We process basins from highest spill elevation to lowest (single pass).
fn calculate_water_levels(basins: &mut [Basin], climate_ratio: f32) {
    if basins.is_empty() {
        return;
    }

    let n = basins.len();

    // Sort basin indices by spill elevation descending (highest first)
    // Higher basins overflow into lower ones, so process high to low
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        basins[b]
            .spill_elevation
            .partial_cmp(&basins[a].spill_elevation)
            .unwrap()
    });

    // Track overflow received by each basin from upstream
    let mut overflow_catchment = vec![0.0f32; n];

    for &i in &order {
        let basin = &basins[i];

        if basin.cells.is_empty() || basin.sorted_elevations.is_empty() {
            continue;
        }

        // Effective catchment = own catchment + overflow from upstream basins
        let effective_catchment = basin.catchment_area as f32 + overflow_catchment[i];

        // Target surface area (in cells) at equilibrium.
        // Using floor helps suppress tiny "on/off" lakes caused by rounding.
        let target_surface = (effective_catchment * climate_ratio).floor() as usize;
        let capacity = basin.cells.len();

        // Determine water level and potential overflow
        let (water_level, _overflows) = if target_surface == 0 {
            // No water - dry basin
            (basin.bottom_elevation - 1.0, false)
        } else if target_surface >= capacity {
            // Basin overflows - fills to spill point
            // Overflow amount = effective_catchment - catchment_needed_to_fill
            // catchment_needed_to_fill = capacity / climate_ratio
            let overflow_amount = effective_catchment - (capacity as f32 / climate_ratio);

            if let Some(downstream) = basin.overflow_target {
                overflow_catchment[downstream] += overflow_amount.max(0.0);
            }

            (basin.spill_elevation, true)
        } else {
            // Partial fill - find water level from hypsometric curve
            let submerge_idx = target_surface - 1;
            let cell_elev = basin.sorted_elevations[submerge_idx];

            let level = if target_surface < basin.sorted_elevations.len() {
                let next_elev = basin.sorted_elevations[target_surface];
                (cell_elev + next_elev) / 2.0
            } else {
                basin.spill_elevation
            };

            (level, false)
        };

        // Apply minimum depth threshold based on maximum possible depth
        // A basin that's too shallow overall shouldn't form a lake even if it overflows
        let max_depth = basin.spill_elevation - basin.bottom_elevation;
        let final_level = if max_depth < MIN_LAKE_DEPTH {
            basin.bottom_elevation - 1.0 // Basin too shallow to form a lake
        } else {
            water_level
        };

        // We need to mutate, so access again
        basins[i].water_level = final_level;
    }
}

/// Compute drainage direction via steepest descent on filled elevation.
///
/// Each cell drains to its lowest neighbor. This is deterministic and
/// follows the actual gradient, unlike directions from the flood itself.
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

        // Find the neighbor with lowest filled elevation
        let mut best_neighbor: Option<usize> = None;
        let mut best_elev = cell_elev;

        for &neighbor in tessellation.neighbors(cell) {
            let neighbor_elev = filled_elevation[neighbor];
            if neighbor_elev < best_elev {
                best_elev = neighbor_elev;
                best_neighbor = Some(neighbor);
            }
        }

        // On flats (no strictly-lower neighbor), fall back to the priority-flood parent.
        // This produces consistent drainage across filled plateaus and basin floors.
        if best_neighbor.is_none() {
            best_neighbor = flood_parent[cell];
        }

        drainage[cell] = best_neighbor;
    }

    drainage
}

/// Compute flow accumulation using topological sort.
///
/// Each cell contributes 1 unit of "rainfall" that flows downstream.
/// Processes cells in dependency order (all upstream before downstream).
fn compute_flow_accumulation(
    _tessellation: &Tessellation,
    drainage_dir: &[Option<usize>],
) -> Vec<f32> {
    let n = drainage_dir.len();

    // Count how many cells drain into each cell (upstream count)
    let mut upstream_count = vec![0usize; n];
    for downstream in drainage_dir.iter().flatten() {
        upstream_count[*downstream] += 1;
    }

    // Use topological sort: process cells with no remaining upstream dependencies
    let mut flow = vec![1.0f32; n]; // Each cell starts with 1 unit
    let mut remaining_upstream = upstream_count.clone();
    let mut ready: Vec<usize> = (0..n).filter(|&c| upstream_count[c] == 0).collect();

    while let Some(cell) = ready.pop() {
        if let Some(downstream) = drainage_dir[cell] {
            flow[downstream] += flow[cell];
            remaining_upstream[downstream] -= 1;
            if remaining_upstream[downstream] == 0 {
                ready.push(downstream);
            }
        }
    }

    flow
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
