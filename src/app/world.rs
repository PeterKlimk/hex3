use glam::Vec3;
use wgpu::util::DeviceExt;

use hex3::geometry::{
    Material, MeshVertex, SurfaceVertex, UnifiedMesh, UnifiedVertex, VoronoiMesh,
};
use hex3::render::{create_index_buffer, create_vertex_buffer, ElevationVertex};
use hex3::util::Timed;
use hex3::world::{VoronoiBackend, World};

use super::coloring::{
    cell_color_climate, cell_color_elevation, cell_color_feature, cell_color_hydrology,
    cell_color_noise, cell_color_plate, cell_color_terrain, cell_material,
};
use super::view::{ClimateLayer, FeatureLayer, NoiseLayer, RenderMode};
use super::visualization::{
    build_boundary_edge_colors, generate_pole_markers, generate_velocity_arrows,
};

pub const NUM_CELLS: usize = 100000;
pub const LLOYD_ITERATIONS: usize = 5;
pub const NUM_PLATES: usize = hex3::world::NUM_PLATES_DEFAULT;

/// Minimum flow for "all rivers" mode, as fraction of total cells.
/// E.g., 0.0003 means a cell needs 0.03% of total cells draining through it.
const RIVER_MIN_FLOW_FRACTION: f32 = 0.0003;

/// Minimum flow for a river mouth to be a "major outlet", as fraction of total cells.
/// Rivers are traced upstream from outlets exceeding this threshold.
/// E.g., 0.002 means outlet must drain 0.2% of all cells.
const RIVER_OUTLET_FRACTION: f32 = 0.004;

/// Minimum flow to continue tracing a river branch, as fraction of total cells.
/// At confluences, branches above this threshold are followed.
const RIVER_BRANCH_FRACTION: f32 = 0.0006;

// River geometry constants (for triangle strip rendering)
/// Minimum river width (smallest tributaries)
const RIVER_MIN_WIDTH: f32 = 0.003;
/// Maximum river width (major rivers)
const RIVER_MAX_WIDTH: f32 = 0.015;

fn buffer_bytes<T>(items: &[T]) -> usize {
    items.len() * std::mem::size_of::<T>()
}

/// All GPU buffers for world rendering.
/// Simplified: one dynamic colored mesh + specialized buffers for Relief/rivers/overlays.
pub struct WorldBuffers {
    // Dynamic colored mesh (regenerated on mode/layer switch) - used by most modes
    pub colored_vertex_buffer: wgpu::Buffer,
    pub colored_index_buffer: wgpu::Buffer,
    pub num_colored_indices: u32,

    // Edge lines (two variants: default gray, plates with colored boundaries)
    pub edge_vertex_buffer: wgpu::Buffer,
    pub edge_vertex_buffer_plates: wgpu::Buffer,
    pub num_edge_vertices: u32,
    pub num_edge_vertices_plates: u32,

    // Relief mode: unified mesh with materials + elevation + relief edges
    pub unified_vertex_buffer: wgpu::Buffer,
    pub unified_index_buffer: wgpu::Buffer,
    pub num_unified_indices: u32,
    pub relief_edge_vertex_buffer: wgpu::Buffer,
    pub relief_edge_index_buffer: wgpu::Buffer,
    pub num_relief_edge_indices: u32,

    // Rivers (line-based for non-relief, triangle mesh for relief)
    pub river_all_vertex_buffer: wgpu::Buffer,
    pub river_major_vertex_buffer: wgpu::Buffer,
    pub num_river_all_vertices: u32,
    pub num_river_major_vertices: u32,
    pub river_mesh_all_vertex_buffer: wgpu::Buffer,
    pub river_mesh_all_index_buffer: wgpu::Buffer,
    pub river_mesh_major_vertex_buffer: wgpu::Buffer,
    pub river_mesh_major_index_buffer: wgpu::Buffer,
    pub num_river_mesh_all_indices: u32,
    pub num_river_mesh_major_indices: u32,

    // Plate overlays (arrows + pole markers)
    pub arrow_vertex_buffer: wgpu::Buffer,
    pub pole_marker_vertex_buffer: wgpu::Buffer,
    pub pole_marker_index_buffer: wgpu::Buffer,
    pub num_arrow_vertices: u32,
    pub num_pole_marker_indices: u32,
}

impl WorldBuffers {
    /// Get the river buffer and vertex count for the given river mode.
    pub fn river_buffer(&self, mode: super::view::RiverMode) -> Option<(&wgpu::Buffer, u32)> {
        match mode {
            super::view::RiverMode::Off => None,
            super::view::RiverMode::Major => Some((
                &self.river_major_vertex_buffer,
                self.num_river_major_vertices,
            )),
            super::view::RiverMode::All => {
                Some((&self.river_all_vertex_buffer, self.num_river_all_vertices))
            }
        }
    }
}

pub fn create_world_with_options(seed: u64, backend: VoronoiBackend) -> World {
    let _total = Timed::info("Stage 1 (Lithosphere)");
    log::info!(
        "Generating world: seed={}, cells={}, lloyd={}, plates={}, voronoi_backend={}",
        seed,
        NUM_CELLS,
        LLOYD_ITERATIONS,
        NUM_PLATES,
        backend
    );

    let mut world = {
        let _t = Timed::info("Tessellation");
        World::new_with_options(seed, NUM_CELLS, LLOYD_ITERATIONS, backend)
    };

    {
        let _t = Timed::info("Plates");
        world.generate_plates(NUM_PLATES);
    }

    {
        let _t = Timed::info("Crust");
        world.generate_crust();
    }

    {
        let _t = Timed::info("Dynamics");
        world.generate_dynamics();
    }

    {
        let _t = Timed::info("Features");
        world.generate_features();
    }

    {
        let _t = Timed::info("Elevation");
        world.generate_elevation();
    }

    print_world_stats(&world);

    world
}

/// Advance world to Stage 2 (Atmosphere).
pub fn advance_to_stage_2(world: &mut World) {
    let _total = Timed::info("Stage 2 (Atmosphere)");

    {
        let _t = Timed::info("Atmosphere");
        world.generate_atmosphere();
    }

    // Print atmosphere stats
    if let Some(atmosphere) = &world.atmosphere {
        let stats = atmosphere.stats();
        let (mean_wind_delta, max_wind_delta, mean_upper, mean_surface) = {
            let n = atmosphere.wind.len().max(1) as f32;
            let mut sum_delta = 0.0_f32;
            let mut max_delta = 0.0_f32;
            let mut sum_upper = 0.0_f32;
            let mut sum_surface = 0.0_f32;
            for (u, s) in atmosphere.upper_wind.iter().zip(atmosphere.wind.iter()) {
                sum_upper += u.length();
                sum_surface += s.length();
                let d = (*u - *s).length();
                sum_delta += d;
                max_delta = max_delta.max(d);
            }
            (sum_delta / n, max_delta, sum_upper / n, sum_surface / n)
        };
        log::info!(
            "Atmosphere: temp=[{:.2}, {:.2}], mean={:.2}, mean_wind=[{:.2},{:.2}], max_wind=[{:.2},{:.2}], wind_delta=[{:.3},{:.3}], max_uplift={:.2}",
            stats.min_temp,
            stats.max_temp,
            stats.mean_temp,
            mean_upper,
            mean_surface,
            stats.max_upper_wind,
            stats.max_wind,
            mean_wind_delta,
            max_wind_delta,
            stats.max_uplift
        );
    }
}

/// Advance world to Stage 3 (Hydrology).
pub fn advance_to_stage_3(world: &mut World) {
    let _total = Timed::info("Stage 3 (Hydrosphere)");

    {
        let _t = Timed::info("Hydrology");
        world.generate_hydrology();
    }

    // Print hydrology stats
    if let Some(hydrology) = world.active_hydrology() {
        let num_cells = world.num_cells();

        let ocean_cells = (0..num_cells).filter(|&i| hydrology.is_ocean(i)).count();
        let lake_cells = (0..num_cells)
            .filter(|&i| hydrology.is_lake_water(i))
            .count();
        let dry_basin_cells = (0..num_cells)
            .filter(|&i| hydrology.is_dry_basin(i))
            .count();
        let land_cells = (0..num_cells)
            .filter(|&i| !hydrology.is_submerged(i))
            .count();
        let non_ocean_cells = num_cells - ocean_cells;
        let lake_pct = if non_ocean_cells > 0 {
            100.0 * lake_cells as f32 / non_ocean_cells as f32
        } else {
            0.0
        };

        let cells_with_drainage = (0..num_cells)
            .filter(|&i| hydrology.downstream(i).is_some())
            .count();

        // Compute resolution-independent thresholds
        let river_min_flow = (num_cells as f32 * RIVER_MIN_FLOW_FRACTION).max(1.0);

        let river_cells = hydrology.river_cells(river_min_flow);
        let max_flow = hydrology
            .flow_accumulation
            .iter()
            .copied()
            .fold(0.0f32, f32::max);

        log::info!(
            "Hydrology: ocean={}, land={}, lakes={} ({:.1}%), basins={} ({} dry)",
            ocean_cells,
            land_cells,
            lake_cells,
            lake_pct,
            hydrology.basins.len(),
            dry_basin_cells
        );
        log::info!(
            "Rivers: drainage={} cells, rivers={} (flow>={:.0}), max_flow={:.0}",
            cells_with_drainage,
            river_cells.len(),
            river_min_flow,
            max_flow
        );
        if let Some(fine) = &world.fine {
            log::info!(
                "Fine mesh: coarse_cells={}, fine_cells={}, density_ratio={:.1}:1",
                world.tessellation.num_cells(),
                fine.tessellation.num_cells(),
                fine.achieved_density_ratio
            );
        }
    }
}

/// Compute resolution-independent river thresholds from cell count.
/// River rendering color - muted blue matching lake/water colors.
const RIVER_COLOR: Vec3 = Vec3::new(0.15, 0.35, 0.60);

/// Which subset of river segments to render.
#[derive(Clone, Copy)]
enum RiverSet {
    /// All drainage above the minimum flow threshold.
    All,
    /// Only cells belonging to major rivers (traced upstream from large outlets).
    Major,
}

/// Per-cell mask of which cells emit a river segment for the given set.
fn river_cell_mask(
    hydrology: &hex3::world::Hydrology,
    num_cells: usize,
    set: RiverSet,
) -> Vec<bool> {
    match set {
        RiverSet::All => {
            let (min_flow, _, _) = river_thresholds(num_cells);
            (0..num_cells)
                .map(|i| hydrology.flow_accumulation[i] >= min_flow && !hydrology.is_submerged(i))
                .collect()
        }
        RiverSet::Major => {
            let (_, outlet_threshold, branch_threshold) = river_thresholds(num_cells);
            hydrology.compute_major_river_cells(outlet_threshold, branch_threshold)
        }
    }
}

fn river_thresholds(num_cells: usize) -> (f32, f32, f32) {
    let min_flow = (num_cells as f32 * RIVER_MIN_FLOW_FRACTION).max(1.0);
    let outlet_threshold = (num_cells as f32 * RIVER_OUTLET_FRACTION).max(1.0);
    let branch_threshold = (num_cells as f32 * RIVER_BRANCH_FRACTION).max(1.0);
    (min_flow, outlet_threshold, branch_threshold)
}

fn mode_uses_fine_mesh(world: &World, mode: RenderMode) -> bool {
    world.fine.is_some()
        && matches!(
            mode,
            RenderMode::Relief
                | RenderMode::Terrain
                | RenderMode::Elevation
                | RenderMode::Hydrology
                | RenderMode::Climate
        )
}

/// Generate a colored mesh for a specific render mode and layer settings.
/// This is fast (~5-10ms for 80k cells) and called on mode/layer switch.
pub fn generate_colored_mesh(
    device: &wgpu::Device,
    world: &World,
    mode: RenderMode,
    noise_layer: NoiseLayer,
    feature_layer: FeatureLayer,
    climate_layer: ClimateLayer,
) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let use_fine = mode_uses_fine_mesh(world, mode);
    let voronoi = if use_fine {
        &world.active_tessellation().voronoi
    } else {
        &world.tessellation.voronoi
    };

    let mesh = match mode {
        RenderMode::Relief | RenderMode::Terrain => {
            if use_fine {
                VoronoiMesh::from_voronoi_shared_vertices(voronoi, |i| cell_color_terrain(world, i))
            } else {
                VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_terrain(world, i))
            }
        }
        RenderMode::Elevation => {
            if use_fine {
                VoronoiMesh::from_voronoi_shared_vertices(voronoi, |i| {
                    cell_color_elevation(world, i)
                })
            } else {
                VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_elevation(world, i))
            }
        }
        RenderMode::Plates => {
            VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_plate(world, i))
        }
        RenderMode::Noise => VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
            cell_color_noise(world, i, noise_layer)
        }),
        RenderMode::Hydrology => {
            if use_fine {
                VoronoiMesh::from_voronoi_shared_vertices(voronoi, |i| {
                    cell_color_hydrology(world, i)
                })
            } else {
                VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_hydrology(world, i))
            }
        }
        RenderMode::Features => VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
            cell_color_feature(world, i, feature_layer)
        }),
        RenderMode::Climate => {
            if use_fine {
                VoronoiMesh::from_voronoi_shared_vertices(voronoi, |i| {
                    cell_color_climate(world, i, climate_layer)
                })
            } else {
                VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
                    cell_color_climate(world, i, climate_layer)
                })
            }
        }
    };

    let vertex_buffer = create_vertex_buffer(device, &mesh.vertices, "colored_vertex");
    let index_buffer = create_index_buffer(device, &mesh.indices, "colored_index");
    let num_indices = mesh.indices.len() as u32;
    if use_fine {
        let vertex_bytes = buffer_bytes(&mesh.vertices);
        let index_bytes = buffer_bytes(&mesh.indices);
        log::info!(
            "fine mesh GPU colored mesh ({}): vertices={}, indices={}, vertex_bytes={:.1} MiB, index_bytes={:.1} MiB, total={:.1} MiB",
            mode.name(),
            mesh.vertices.len(),
            mesh.indices.len(),
            vertex_bytes as f64 / 1_048_576.0,
            index_bytes as f64 / 1_048_576.0,
            (vertex_bytes + index_bytes) as f64 / 1_048_576.0,
        );
    }

    (vertex_buffer, index_buffer, num_indices)
}

/// Generate GPU buffers from a World.
/// Creates one dynamic colored mesh (initially Terrain mode) plus specialized buffers.
pub fn generate_world_buffers(device: &wgpu::Device, world: &World) -> WorldBuffers {
    let use_fine = world.fine.is_some();
    let voronoi = &world.active_tessellation().voronoi;
    let elevation = world.active_elevation().unwrap();

    let _t = Timed::debug("Build world buffers");

    // Initial colored mesh (Terrain mode - will be regenerated on mode switch)
    let (colored_vertex_buffer, colored_index_buffer, num_colored_indices) = generate_colored_mesh(
        device,
        world,
        RenderMode::Terrain,
        NoiseLayer::Combined,
        FeatureLayer::Trench,
        ClimateLayer::Temperature,
    );

    // Unified mesh with material-aware lighting for Relief mode
    let elevation_for_cell = |i| {
        if let Some(hydrology) = world.active_hydrology() {
            if hydrology.is_ocean(i) {
                return 0.0;
            }
            if hydrology.is_lake_water(i) {
                return hydrology.basin(i).map(|b| b.water_level).unwrap_or(0.0);
            }
        }
        elevation.values[i].max(0.0)
    };

    let unified_mesh = if use_fine {
        UnifiedMesh::from_voronoi_shared_vertices(
            voronoi,
            |i| cell_color_terrain(world, i),
            |i| cell_material(world, i),
            elevation_for_cell,
        )
    } else {
        UnifiedMesh::from_voronoi_with_elevation(
            voronoi,
            |i| cell_color_terrain(world, i),
            |i| cell_material(world, i),
            elevation_for_cell,
        )
    };

    // Edge lines: default gray + plates with colored boundaries
    let edge_color = Vec3::new(0.35, 0.35, 0.35);
    let edge_vertices_default = if use_fine {
        Vec::new()
    } else {
        VoronoiMesh::edge_lines_with_colors(voronoi, |_, _| edge_color)
    };

    let boundary_edge_colors = build_boundary_edge_colors(world);
    let edge_vertices_plates =
        VoronoiMesh::edge_lines_with_colors(&world.tessellation.voronoi, |a, b| {
            let key = if a < b { (a, b) } else { (b, a) };
            boundary_edge_colors
                .get(&key)
                .copied()
                .unwrap_or(edge_color)
        });

    // Relief edges with elevation — indexed line list (shared vertices). This
    // is the real Voronoi wireframe at both resolutions; indexing keeps the
    // fine mesh (~3 edges/cell at 2.5M cells) to a tractable buffer size.
    let (relief_edge_vertices, relief_edge_indices) =
        VoronoiMesh::edge_lines_indexed_with_elevation(
            voronoi,
            edge_color,
            elevation_for_cell,
            |i| cell_material(world, i),
        );
    if use_fine {
        log::info!(
            "fine mesh relief wireframe: vertices={}, edges={}, vertex_bytes={:.1} MiB, index_bytes={:.1} MiB",
            relief_edge_vertices.len(),
            relief_edge_indices.len() / 2,
            buffer_bytes(&relief_edge_vertices) as f64 / 1_048_576.0,
            buffer_bytes(&relief_edge_indices) as f64 / 1_048_576.0,
        );
    }

    // Plate overlays
    let arrows = generate_velocity_arrows(world);
    let pole_markers = generate_pole_markers(world);

    let arrow_vertices: Vec<MeshVertex> = arrows
        .iter()
        .flat_map(|&(start, end, color)| {
            [
                MeshVertex::new(start, start, color),
                MeshVertex::new(end, end, color),
            ]
        })
        .collect();

    let pole_marker_vertices: Vec<MeshVertex> = pole_markers
        .iter()
        .map(|&(pos, normal, color)| MeshVertex::new(pos, normal, color))
        .collect();
    let pole_marker_indices: Vec<u32> = (0..pole_marker_vertices.len() as u32).collect();

    log::debug!(
        "Overlays: {} arrows, {} pole markers, {} boundary edges",
        arrows.len() / 3,
        pole_markers.len() / 3,
        boundary_edge_colors.len()
    );

    // Rivers
    let river_all_vertices = generate_river_vertices(world, RiverSet::All);
    let river_major_vertices = generate_river_vertices(world, RiverSet::Major);
    let river_mesh_all = generate_river_mesh(world, RiverSet::All);
    let river_mesh_major = generate_river_mesh(world, RiverSet::Major);

    if !river_all_vertices.is_empty() {
        log::debug!(
            "River segments: {} line, {} triangles (major: {} line, {} triangles)",
            river_all_vertices.len() / 2,
            river_mesh_all.indices.len() / 3,
            river_major_vertices.len() / 2,
            river_mesh_major.indices.len() / 3,
        );
    }

    if use_fine {
        let unified_vertex_bytes = buffer_bytes(&unified_mesh.vertices);
        let unified_index_bytes = buffer_bytes(&unified_mesh.indices);
        let river_vertex_bytes =
            buffer_bytes(&river_mesh_all.vertices) + buffer_bytes(&river_mesh_major.vertices);
        let river_index_bytes =
            buffer_bytes(&river_mesh_all.indices) + buffer_bytes(&river_mesh_major.indices);
        log::info!(
            "fine mesh GPU relief mesh: vertices={}, indices={}, vertex_bytes={:.1} MiB, index_bytes={:.1} MiB, total={:.1} MiB",
            unified_mesh.vertices.len(),
            unified_mesh.indices.len(),
            unified_vertex_bytes as f64 / 1_048_576.0,
            unified_index_bytes as f64 / 1_048_576.0,
            (unified_vertex_bytes + unified_index_bytes) as f64 / 1_048_576.0,
        );
        log::info!(
            "fine mesh GPU river meshes: all_indices={}, major_indices={}, vertex_bytes={:.1} MiB, index_bytes={:.1} MiB, total={:.1} MiB",
            river_mesh_all.indices.len(),
            river_mesh_major.indices.len(),
            river_vertex_bytes as f64 / 1_048_576.0,
            river_index_bytes as f64 / 1_048_576.0,
            (river_vertex_bytes + river_index_bytes) as f64 / 1_048_576.0,
        );
    }

    drop(_t);

    WorldBuffers {
        // Dynamic colored mesh
        colored_vertex_buffer,
        colored_index_buffer,
        num_colored_indices,

        // Edges
        edge_vertex_buffer: create_vertex_buffer(device, &edge_vertices_default, "edge_vertex"),
        edge_vertex_buffer_plates: create_vertex_buffer(
            device,
            &edge_vertices_plates,
            "edge_vertex_plates",
        ),
        num_edge_vertices: edge_vertices_default.len() as u32,
        num_edge_vertices_plates: edge_vertices_plates.len() as u32,

        // Relief mode
        unified_vertex_buffer: create_vertex_buffer(
            device,
            &unified_mesh.vertices,
            "unified_vertex",
        ),
        unified_index_buffer: create_index_buffer(device, &unified_mesh.indices, "unified_index"),
        num_unified_indices: unified_mesh.indices.len() as u32,
        relief_edge_vertex_buffer: create_vertex_buffer(
            device,
            &relief_edge_vertices,
            "relief_edge_vertex",
        ),
        relief_edge_index_buffer: create_index_buffer(
            device,
            &relief_edge_indices,
            "relief_edge_index",
        ),
        num_relief_edge_indices: relief_edge_indices.len() as u32,

        // Rivers
        river_all_vertex_buffer: create_vertex_buffer(
            device,
            &river_all_vertices,
            "river_all_vertex",
        ),
        river_major_vertex_buffer: create_vertex_buffer(
            device,
            &river_major_vertices,
            "river_major_vertex",
        ),
        num_river_all_vertices: river_all_vertices.len() as u32,
        num_river_major_vertices: river_major_vertices.len() as u32,
        river_mesh_all_vertex_buffer: create_vertex_buffer(
            device,
            &river_mesh_all.vertices,
            "river_mesh_all_vertex",
        ),
        river_mesh_all_index_buffer: create_index_buffer(
            device,
            &river_mesh_all.indices,
            "river_mesh_all_index",
        ),
        river_mesh_major_vertex_buffer: create_vertex_buffer(
            device,
            &river_mesh_major.vertices,
            "river_mesh_major_vertex",
        ),
        river_mesh_major_index_buffer: create_index_buffer(
            device,
            &river_mesh_major.indices,
            "river_mesh_major_index",
        ),
        num_river_mesh_all_indices: river_mesh_all.indices.len() as u32,
        num_river_mesh_major_indices: river_mesh_major.indices.len() as u32,

        // Plate overlays
        arrow_vertex_buffer: create_vertex_buffer(device, &arrow_vertices, "arrow_vertex"),
        pole_marker_vertex_buffer: create_vertex_buffer(
            device,
            &pole_marker_vertices,
            "pole_marker_vertex",
        ),
        pole_marker_index_buffer: create_index_buffer(
            device,
            &pole_marker_indices,
            "pole_marker_index",
        ),
        num_arrow_vertices: arrow_vertices.len() as u32,
        num_pole_marker_indices: pole_marker_indices.len() as u32,
    }
}

/// Generate elevation mesh for rendering to elevation map texture.
/// Returns (vertex_buffer, index_buffer, num_indices).
///
/// Uses the same coastal handling as the unified mesh:
/// - Water cells are flat at their water level
/// - Land cells at water boundary use water level (smooth coast transition)
/// - Interior land uses averaged elevation from adjacent cells
pub fn generate_elevation_mesh_buffers(
    device: &wgpu::Device,
    world: &World,
) -> (wgpu::Buffer, wgpu::Buffer, u32) {
    let tessellation = world.active_tessellation();
    let voronoi = &tessellation.voronoi;
    let elevation = world.active_elevation().unwrap();

    // Step 1: Compute per-cell elevation and water status
    let cell_elevations: Vec<f32> = (0..voronoi.num_cells())
        .map(|cell_idx| {
            if let Some(hydrology) = world.active_hydrology() {
                if hydrology.is_ocean(cell_idx) {
                    0.0
                } else if hydrology.is_lake_water(cell_idx) {
                    hydrology
                        .basin(cell_idx)
                        .map(|b| b.water_level)
                        .unwrap_or(0.0)
                } else {
                    elevation.values[cell_idx].max(0.0)
                }
            } else {
                elevation.values[cell_idx].max(0.0)
            }
        })
        .collect();

    let cell_is_water: Vec<bool> = (0..voronoi.num_cells())
        .map(|cell_idx| {
            if let Some(hydrology) = world.active_hydrology() {
                hydrology.is_ocean(cell_idx) || hydrology.is_lake_water(cell_idx)
            } else {
                elevation.values[cell_idx] <= 0.0
            }
        })
        .collect();

    // Step 2: For each vertex, track water level and land elevation statistics
    let mut vertex_land_sum = vec![0.0f32; voronoi.vertices.len()];
    let mut vertex_land_count = vec![0u32; voronoi.vertices.len()];
    let mut vertex_water_level = vec![None::<f32>; voronoi.vertices.len()];

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        let elev = cell_elevations[cell_idx];
        let is_water = cell_is_water[cell_idx];

        for &vertex_idx in cell.vertex_indices {
            let vi = vertex_idx as usize;
            if is_water {
                // Track water level (use max in case of adjacent lakes at different levels)
                vertex_water_level[vi] = Some(
                    vertex_water_level[vi]
                        .map(|wl| wl.max(elev))
                        .unwrap_or(elev),
                );
            } else {
                // Accumulate land elevation for averaging
                vertex_land_sum[vi] += elev;
                vertex_land_count[vi] += 1;
            }
        }
    }

    // Step 3: Compute final per-vertex elevations with water boundary handling
    let vertex_elevations: Vec<f32> = (0..voronoi.vertices.len())
        .map(|v| {
            let water_level = vertex_water_level[v];
            let land_count = vertex_land_count[v];

            match (water_level, land_count) {
                // All water: use water level
                (Some(wl), 0) => wl,
                // All land: average land elevations
                (None, n) if n > 0 => vertex_land_sum[v] / n as f32,
                // Mixed (land touching water): use max of land average and water level
                (Some(wl), n) if n > 0 => {
                    let land_avg = vertex_land_sum[v] / n as f32;
                    land_avg.max(wl)
                }
                _ => 0.0,
            }
        })
        .collect();

    // Step 4: Build a shared-vertex mesh with proper coastal elevation handling.
    let mut vertices = Vec::with_capacity(voronoi.vertices.len());
    for (vi, &pos) in voronoi.vertices.iter().enumerate() {
        vertices.push(ElevationVertex {
            position: [pos.x, pos.y, pos.z],
            elevation: vertex_elevations[vi],
        });
    }
    let mut indices = Vec::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        if cell.len() < 3 {
            continue;
        }

        // Fan triangulation
        let n = cell.vertex_indices.len();
        for i in 1..n - 1 {
            indices.push(cell.vertex_indices[0]);
            indices.push(cell.vertex_indices[i]);
            indices.push(cell.vertex_indices[i + 1]);
        }
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elevation_mesh_vertex_buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("elevation_mesh_index_buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    (vertex_buffer, index_buffer, indices.len() as u32)
}

/// Generate line-based river vertices (used outside Relief mode).
/// Alpha encodes flow magnitude on a log scale.
fn generate_river_vertices(world: &World, set: RiverSet) -> Vec<SurfaceVertex> {
    let Some(hydrology) = world.active_hydrology() else {
        return Vec::new();
    };
    let elevation = world.active_elevation().unwrap();
    let tessellation = world.active_tessellation();

    let include = river_cell_mask(hydrology, tessellation.num_cells(), set);

    // Find max flow for normalization
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let log_max = max_flow.ln();

    let mut vertices = Vec::new();

    for (cell_idx, &included) in include.iter().enumerate() {
        if !included {
            continue;
        }

        let Some(downstream_idx) = hydrology.downstream(cell_idx) else {
            continue;
        };

        let (start_pos, end_pos, start_elev, end_elev) =
            river_segment_geometry(tessellation, elevation, hydrology, cell_idx, downstream_idx);

        // Alpha based on logarithmic flow
        let flow = hydrology.flow_accumulation[cell_idx];
        let alpha = 0.15 + 0.55 * (flow.ln() / log_max).clamp(0.0, 1.0);

        vertices.push(SurfaceVertex::new(
            start_pos,
            start_elev,
            RIVER_COLOR,
            alpha,
        ));
        vertices.push(SurfaceVertex::new(end_pos, end_elev, RIVER_COLOR, alpha));
    }

    // Add lake outflow rivers (from overflowing lakes)
    generate_lake_outflow_vertices(world, &mut vertices, RIVER_COLOR);

    vertices
}

/// Helper to compute geometry for a river segment (line-based rendering).
fn river_segment_geometry(
    tessellation: &hex3::world::Tessellation,
    elevation: &hex3::world::Elevation,
    hydrology: &hex3::world::Hydrology,
    cell_idx: usize,
    downstream_idx: usize,
) -> (Vec3, Vec3, f32, f32) {
    let start_center = tessellation.cell_center(cell_idx);
    let end_center = tessellation.cell_center(downstream_idx);

    let start_elev = elevation.values[cell_idx];
    let end_elev = if hydrology.is_submerged(downstream_idx) {
        if hydrology.is_ocean(downstream_idx) {
            0.0
        } else {
            hydrology
                .basin(downstream_idx)
                .map(|b| b.water_level)
                .unwrap_or(0.0)
        }
    } else {
        elevation.values[downstream_idx]
    };

    let end_pos = if hydrology.is_submerged(downstream_idx) {
        ((start_center + end_center) / 2.0).normalize()
    } else {
        end_center
    };

    (start_center, end_pos, start_elev, end_elev)
}

/// Generate vertices for lake outflow rivers (from overflowing lakes).
/// These are added to the existing vertices vector.
fn generate_lake_outflow_vertices(
    world: &World,
    vertices: &mut Vec<SurfaceVertex>,
    river_color: Vec3,
) {
    let Some(hydrology) = world.active_hydrology() else {
        return;
    };
    let elevation = world.active_elevation().unwrap();
    let tessellation = world.active_tessellation();

    // Lake outflows are significant rivers - use high alpha
    let outflow_alpha = 0.7;

    for (_basin_idx, path) in hydrology.lake_outflow_paths() {
        // Generate segments along the outflow path
        for window in path.windows(2) {
            let cell_idx = window[0];
            let downstream_idx = window[1];

            let (start_pos, end_pos, start_elev, end_elev) = river_segment_geometry(
                tessellation,
                elevation,
                hydrology,
                cell_idx,
                downstream_idx,
            );

            vertices.push(SurfaceVertex::new(
                start_pos,
                start_elev,
                river_color,
                outflow_alpha,
            ));
            vertices.push(SurfaceVertex::new(
                end_pos,
                end_elev,
                river_color,
                outflow_alpha,
            ));
        }

        // Add final segment to water (if path ends at land cell adjacent to water)
        if let Some(&last_cell) = path.last() {
            if let Some(downstream_idx) = hydrology.downstream(last_cell) {
                if hydrology.is_submerged(downstream_idx) {
                    let (start_pos, end_pos, start_elev, end_elev) = river_segment_geometry(
                        tessellation,
                        elevation,
                        hydrology,
                        last_cell,
                        downstream_idx,
                    );

                    vertices.push(SurfaceVertex::new(
                        start_pos,
                        start_elev,
                        river_color,
                        outflow_alpha,
                    ));
                    vertices.push(SurfaceVertex::new(
                        end_pos,
                        end_elev,
                        river_color,
                        outflow_alpha,
                    ));
                }
            }
        }
    }
}

// =============================================================================
// Triangle-based river mesh generation (Phase 2)
// =============================================================================

/// River mesh data: vertices and indices for triangle rendering.
pub struct RiverMesh {
    pub vertices: Vec<UnifiedVertex>,
    pub indices: Vec<u32>,
}

/// Convert flow to river width using sqrt scaling.
fn flow_to_width(flow: f32, max_flow: f32) -> f32 {
    let t = (flow / max_flow).sqrt();
    RIVER_MIN_WIDTH + t * (RIVER_MAX_WIDTH - RIVER_MIN_WIDTH)
}

/// Generate a triangle-strip river mesh (used in Relief mode).
/// Each segment is a quad (2 triangles) with width based on flow.
fn generate_river_mesh(world: &World, set: RiverSet) -> RiverMesh {
    let Some(hydrology) = world.active_hydrology() else {
        return RiverMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    };
    let elevation = world.active_elevation().unwrap();
    let tessellation = world.active_tessellation();

    let include = river_cell_mask(hydrology, tessellation.num_cells(), set);

    // Find max flow for width normalization
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .copied()
        .fold(0.0f32, f32::max);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (cell_idx, &included) in include.iter().enumerate() {
        if !included {
            continue;
        }

        let Some(downstream_idx) = hydrology.downstream(cell_idx) else {
            continue;
        };

        let flow = hydrology.flow_accumulation[cell_idx];
        generate_river_segment_quad(
            tessellation,
            elevation,
            hydrology,
            cell_idx,
            downstream_idx,
            flow,
            max_flow,
            RIVER_COLOR,
            &mut vertices,
            &mut indices,
        );
    }

    // Add lake outflow rivers
    generate_lake_outflow_quads(world, max_flow, RIVER_COLOR, &mut vertices, &mut indices);

    RiverMesh { vertices, indices }
}

/// Generate a quad (2 triangles) for a river segment.
/// Positions are on the unit sphere; shader applies displacement based on elevation.
#[allow(clippy::too_many_arguments)]
fn generate_river_segment_quad(
    tessellation: &hex3::world::Tessellation,
    elevation: &hex3::world::Elevation,
    hydrology: &hex3::world::Hydrology,
    cell_idx: usize,
    downstream_idx: usize,
    flow: f32,
    max_flow: f32,
    color: Vec3,
    vertices: &mut Vec<UnifiedVertex>,
    indices: &mut Vec<u32>,
) {
    let start_center = tessellation.cell_center(cell_idx);
    let end_center = tessellation.cell_center(downstream_idx);

    // Use simulation elevation (without micro noise) - shader adds micro noise
    let start_elev = elevation.values[cell_idx];
    let end_elev = if hydrology.is_submerged(downstream_idx) {
        if hydrology.is_ocean(downstream_idx) {
            0.0
        } else {
            hydrology
                .basin(downstream_idx)
                .map(|b| b.water_level)
                .unwrap_or(0.0)
        }
    } else {
        elevation.values[downstream_idx]
    };

    // End position: stop at water's edge if downstream is submerged
    let end_pos = if hydrology.is_submerged(downstream_idx) {
        ((start_center + end_center) / 2.0).normalize()
    } else {
        end_center
    };

    // Normals (on unit sphere, position = normal for lighting)
    let n1 = start_center;
    let n2 = end_pos;

    // Tangent along river flow (this segment)
    let tangent = (end_pos - start_center).normalize();

    // Bitangent perpendicular to flow, in tangent plane of sphere
    let bitangent1 = tangent.cross(n1).normalize();
    let bitangent2 = tangent.cross(n2).normalize();

    // Width based on flow
    let width = flow_to_width(flow, max_flow);
    let half_width = width / 2.0;

    // Four corners of the quad for this segment (on unit sphere, undisplaced)
    // Shader will apply elevation-based displacement
    let p0 = (start_center + bitangent1 * half_width).normalize();
    let p1 = (start_center - bitangent1 * half_width).normalize();
    let p2 = (end_pos + bitangent2 * half_width).normalize();
    let p3 = (end_pos - bitangent2 * half_width).normalize();

    // Add segment vertices with elevation - shader does displacement
    let base_idx = vertices.len() as u32;
    vertices.push(UnifiedVertex::new(
        p0,
        n1,
        color,
        start_elev,
        Material::River,
    ));
    vertices.push(UnifiedVertex::new(
        p1,
        n1,
        color,
        start_elev,
        Material::River,
    ));
    vertices.push(UnifiedVertex::new(p2, n2, color, end_elev, Material::River));
    vertices.push(UnifiedVertex::new(p3, n2, color, end_elev, Material::River));

    // Two triangles with CCW winding when viewed from outside sphere
    indices.push(base_idx);
    indices.push(base_idx + 2);
    indices.push(base_idx + 1);

    indices.push(base_idx + 1);
    indices.push(base_idx + 2);
    indices.push(base_idx + 3);

    // Add joint triangles if there's a downstream segment (fills gap at bend)
    if !hydrology.is_submerged(downstream_idx) {
        if let Some(next_downstream_idx) = hydrology.downstream(downstream_idx) {
            let next_center = tessellation.cell_center(next_downstream_idx);
            let next_tangent = (next_center - end_center).normalize();
            let next_bitangent = next_tangent.cross(n2).normalize();

            // Next segment's start vertices at end_pos
            let next_width = flow_to_width(hydrology.flow_accumulation[downstream_idx], max_flow);
            let next_half_width = next_width / 2.0;

            let q0 = (end_pos + next_bitangent * next_half_width).normalize();
            let q1 = (end_pos - next_bitangent * next_half_width).normalize();

            // Add joint vertices
            let joint_base = vertices.len() as u32;
            vertices.push(UnifiedVertex::new(q0, n2, color, end_elev, Material::River));
            vertices.push(UnifiedVertex::new(q1, n2, color, end_elev, Material::River));

            // Fill the gap with triangles
            indices.push(base_idx + 2);
            indices.push(joint_base);
            indices.push(base_idx + 3);

            indices.push(base_idx + 3);
            indices.push(joint_base);
            indices.push(joint_base + 1);
        }
    }
}

/// Generate quads for lake outflow rivers.
fn generate_lake_outflow_quads(
    world: &World,
    max_flow: f32,
    color: Vec3,
    vertices: &mut Vec<UnifiedVertex>,
    indices: &mut Vec<u32>,
) {
    let Some(hydrology) = world.active_hydrology() else {
        return;
    };
    let elevation = world.active_elevation().unwrap();
    let tessellation = world.active_tessellation();

    // Lake outflows use a high flow value for width
    let outflow_flow = max_flow * 0.5;

    for (_basin_idx, path) in hydrology.lake_outflow_paths() {
        for window in path.windows(2) {
            let cell_idx = window[0];
            let downstream_idx = window[1];

            generate_river_segment_quad(
                tessellation,
                elevation,
                hydrology,
                cell_idx,
                downstream_idx,
                outflow_flow,
                max_flow,
                color,
                vertices,
                indices,
            );
        }

        // Final segment to water
        if let Some(&last_cell) = path.last() {
            if let Some(downstream_idx) = hydrology.downstream(last_cell) {
                if hydrology.is_submerged(downstream_idx) {
                    generate_river_segment_quad(
                        tessellation,
                        elevation,
                        hydrology,
                        last_cell,
                        downstream_idx,
                        outflow_flow,
                        max_flow,
                        color,
                        vertices,
                        indices,
                    );
                }
            }
        }
    }
}

fn print_world_stats(world: &World) {
    let elevation = world.elevation.as_ref().unwrap();
    let num_cells = world.num_cells();

    let water_count = elevation.values.iter().filter(|&&e| e < 0.0).count();
    let water_pct = 100.0 * water_count as f32 / num_cells as f32;
    let avg_elevation: f32 = elevation.values.iter().sum::<f32>() / num_cells as f32;
    let min_elevation = elevation
        .values
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_elevation = elevation
        .values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let continental_pct = 100.0
        * world
            .crust
            .as_ref()
            .map(|c| c.continental_fraction())
            .unwrap_or(0.0);

    log::info!(
        "Stats: cells={}, water={:.1}%, continental={:.1}%, elev=[{:.3}, {:.3}], avg={:.3}",
        num_cells,
        water_pct,
        continental_pct,
        min_elevation,
        max_elevation,
        avg_elevation
    );
}
