use glam::Vec3;
use wgpu::util::DeviceExt;

use hex3::geometry::{
    Material, MeshVertex, SurfaceVertex, UnifiedMesh, UnifiedVertex, VoronoiMesh,
};
use hex3::render::{create_index_buffer, create_vertex_buffer, ElevationVertex};
use hex3::util::Timed;
use hex3::world::{PlateType, World};

use super::coloring::{
    cell_color_climate, cell_color_elevation, cell_color_feature, cell_color_hydrology,
    cell_color_noise, cell_color_plate, cell_color_terrain, cell_material,
};
use super::view::{ClimateLayer, FeatureLayer, NoiseLayer, RenderMode};
use super::visualization::{
    build_boundary_edge_colors, generate_pole_markers, generate_velocity_arrows,
};

pub const NUM_CELLS: usize = 80000;
pub const LLOYD_ITERATIONS: usize = 5;
pub const NUM_PLATES: usize = 14;

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
    pub num_relief_edge_vertices: u32,

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

/// Generate a new world (Stage 1: Lithosphere).
pub fn create_world(seed: u64) -> World {
    create_world_with_options(seed, false)
}

pub fn create_world_with_options(seed: u64, gpu_voronoi: bool) -> World {
    let _total = Timed::info("Stage 1 (Lithosphere)");
    log::info!(
        "Generating world: seed={}, cells={}, lloyd={}, plates={}, gpu_voronoi={}",
        seed,
        NUM_CELLS,
        LLOYD_ITERATIONS,
        NUM_PLATES,
        gpu_voronoi
    );

    let mut world = {
        let _t = Timed::info("Tessellation");
        World::new_with_options(seed, NUM_CELLS, LLOYD_ITERATIONS, gpu_voronoi)
    };

    {
        let _t = Timed::info("Plates");
        world.generate_plates(NUM_PLATES);
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
    if let Some(hydrology) = &world.hydrology {
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
    }
}

/// Compute resolution-independent river thresholds from cell count.
fn river_thresholds(num_cells: usize) -> (f32, f32, f32) {
    let min_flow = (num_cells as f32 * RIVER_MIN_FLOW_FRACTION).max(1.0);
    let outlet_threshold = (num_cells as f32 * RIVER_OUTLET_FRACTION).max(1.0);
    let branch_threshold = (num_cells as f32 * RIVER_BRANCH_FRACTION).max(1.0);
    (min_flow, outlet_threshold, branch_threshold)
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
    let voronoi = &world.tessellation.voronoi;

    let mesh = match mode {
        RenderMode::Relief | RenderMode::Terrain => {
            VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_terrain(world, i))
        }
        RenderMode::Elevation => {
            VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_elevation(world, i))
        }
        RenderMode::Plates => {
            VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_plate(world, i))
        }
        RenderMode::Noise => VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
            cell_color_noise(world, i, noise_layer)
        }),
        RenderMode::Hydrology => {
            VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_hydrology(world, i))
        }
        RenderMode::Features => VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
            cell_color_feature(world, i, feature_layer)
        }),
        RenderMode::Climate => VoronoiMesh::from_voronoi_with_colors(voronoi, |i| {
            cell_color_climate(world, i, climate_layer)
        }),
    };

    let vertex_buffer = create_vertex_buffer(device, &mesh.vertices, "colored_vertex");
    let index_buffer = create_index_buffer(device, &mesh.indices, "colored_index");
    let num_indices = mesh.indices.len() as u32;

    (vertex_buffer, index_buffer, num_indices)
}

/// Generate GPU buffers from a World.
/// Creates one dynamic colored mesh (initially Terrain mode) plus specialized buffers.
pub fn generate_world_buffers(device: &wgpu::Device, world: &World) -> WorldBuffers {
    let voronoi = &world.tessellation.voronoi;
    let elevation = world.elevation.as_ref().unwrap();

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
    let unified_mesh = UnifiedMesh::from_voronoi_with_elevation(
        voronoi,
        |i| cell_color_terrain(world, i),
        |i| cell_material(world, i),
        |i| {
            if let Some(hydrology) = &world.hydrology {
                if hydrology.is_ocean(i) {
                    return 0.0;
                }
                if hydrology.is_lake_water(i) {
                    return hydrology.basin(i).map(|b| b.water_level).unwrap_or(0.0);
                }
            }
            elevation.values[i].max(0.0)
        },
    );

    // Edge lines: default gray + plates with colored boundaries
    let edge_color = Vec3::new(0.35, 0.35, 0.35);
    let edge_vertices_default = VoronoiMesh::edge_lines_with_colors(voronoi, |_, _| edge_color);

    let boundary_edge_colors = build_boundary_edge_colors(world);
    let edge_vertices_plates = VoronoiMesh::edge_lines_with_colors(voronoi, |a, b| {
        let key = if a < b { (a, b) } else { (b, a) };
        boundary_edge_colors
            .get(&key)
            .copied()
            .unwrap_or(edge_color)
    });

    // Relief edges with elevation
    let edge_vertices_relief = VoronoiMesh::edge_lines_with_elevation(
        voronoi,
        |_, _| edge_color,
        |i| {
            if let Some(hydrology) = &world.hydrology {
                if hydrology.is_ocean(i) {
                    return 0.0;
                }
                if hydrology.is_lake_water(i) {
                    return hydrology.basin(i).map(|b| b.water_level).unwrap_or(0.0);
                }
            }
            elevation.values[i].max(0.0)
        },
        |i| cell_material(world, i),
    );

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
    let river_all_vertices = generate_river_vertices_all(world);
    let river_major_vertices = generate_river_vertices_major(world);
    let river_mesh_all = generate_river_mesh_all(world);
    let river_mesh_major = generate_river_mesh_major(world);

    if !river_all_vertices.is_empty() {
        log::debug!(
            "River segments: {} line, {} triangles (major: {} line, {} triangles)",
            river_all_vertices.len() / 2,
            river_mesh_all.indices.len() / 3,
            river_major_vertices.len() / 2,
            river_mesh_major.indices.len() / 3,
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
            &edge_vertices_relief,
            "relief_edge_vertex",
        ),
        num_relief_edge_vertices: edge_vertices_relief.len() as u32,

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
    let voronoi = &world.tessellation.voronoi;
    let elevation = world.elevation.as_ref().unwrap();

    // Step 1: Compute per-cell elevation and water status
    let cell_elevations: Vec<f32> = (0..voronoi.num_cells())
        .map(|cell_idx| {
            if let Some(hydrology) = &world.hydrology {
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
            if let Some(hydrology) = &world.hydrology {
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
            if is_water {
                // Track water level (use max in case of adjacent lakes at different levels)
                vertex_water_level[vertex_idx] = Some(
                    vertex_water_level[vertex_idx]
                        .map(|wl| wl.max(elev))
                        .unwrap_or(elev),
                );
            } else {
                // Accumulate land elevation for averaging
                vertex_land_sum[vertex_idx] += elev;
                vertex_land_count[vertex_idx] += 1;
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

    // Step 4: Build mesh with proper coastal elevation handling
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for cell_idx in 0..voronoi.num_cells() {
        let cell = voronoi.cell(cell_idx);
        if cell.len() < 3 {
            continue;
        }

        let is_water = cell_is_water[cell_idx];
        let cell_water_level = cell_elevations[cell_idx];

        let base_idx = vertices.len() as u32;

        // Add vertices with proper elevation handling
        for &vertex_idx in cell.vertex_indices {
            let pos = voronoi.vertices[vertex_idx];

            let elev = if is_water {
                // Water cells are flat at their water level
                cell_water_level
            } else if vertex_water_level[vertex_idx].is_some() {
                // Land vertex touching water: use water level for seamless coast
                vertex_water_level[vertex_idx].unwrap()
            } else {
                // Interior land vertex: use averaged elevation
                vertex_elevations[vertex_idx]
            };

            vertices.push(ElevationVertex {
                position: [pos.x, pos.y, pos.z],
                elevation: elev,
            });
        }

        // Fan triangulation
        let n = cell.vertex_indices.len();
        for i in 1..n - 1 {
            indices.push(base_idx);
            indices.push(base_idx + i as u32);
            indices.push(base_idx + (i + 1) as u32);
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

/// Generate river vertices for "all rivers" mode.
/// Shows all drainage with flow >= min_flow threshold, with transparency based on flow.
fn generate_river_vertices_all(world: &World) -> Vec<SurfaceVertex> {
    let Some(hydrology) = &world.hydrology else {
        return Vec::new();
    };
    let elevation = world.elevation.as_ref().unwrap();
    let tessellation = &world.tessellation;

    // Resolution-independent threshold
    let (min_flow, _, _) = river_thresholds(tessellation.num_cells());

    // River color - muted blue matching lake/water colors
    let river_color = Vec3::new(0.15, 0.35, 0.60);

    // Find max flow for normalization
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let log_max = max_flow.ln();

    let mut vertices = Vec::new();

    for cell_idx in 0..tessellation.num_cells() {
        let flow = hydrology.flow_accumulation[cell_idx];

        if flow < min_flow || hydrology.is_submerged(cell_idx) {
            continue;
        }

        let Some(downstream_idx) = hydrology.downstream(cell_idx) else {
            continue;
        };

        let (start_pos, end_pos, start_elev, end_elev) =
            river_segment_geometry(tessellation, elevation, hydrology, cell_idx, downstream_idx);

        // Alpha based on logarithmic flow
        let alpha = 0.15 + 0.55 * (flow.ln() / log_max).clamp(0.0, 1.0);

        vertices.push(SurfaceVertex::new(
            start_pos,
            start_elev,
            river_color,
            alpha,
        ));
        vertices.push(SurfaceVertex::new(end_pos, end_elev, river_color, alpha));
    }

    // Add lake outflow rivers (from overflowing lakes)
    generate_lake_outflow_vertices(world, &mut vertices, river_color);

    vertices
}

/// Generate river vertices for "major rivers" mode.
/// Uses upstream propagation: cells that feed into high-flow rivers are included.
fn generate_river_vertices_major(world: &World) -> Vec<SurfaceVertex> {
    let Some(hydrology) = &world.hydrology else {
        return Vec::new();
    };
    let elevation = world.elevation.as_ref().unwrap();
    let tessellation = &world.tessellation;

    // Resolution-independent thresholds
    let (_, outlet_threshold, branch_threshold) = river_thresholds(tessellation.num_cells());

    // Compute which cells are part of major rivers (traced from outlets upstream)
    let is_major = hydrology.compute_major_river_cells(outlet_threshold, branch_threshold);

    // River color - muted blue matching lake/water colors
    let river_color = Vec3::new(0.15, 0.35, 0.60);

    // Find max flow for normalization
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    let log_max = max_flow.ln();

    let mut vertices = Vec::new();

    for cell_idx in 0..tessellation.num_cells() {
        if !is_major[cell_idx] {
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
            river_color,
            alpha,
        ));
        vertices.push(SurfaceVertex::new(end_pos, end_elev, river_color, alpha));
    }

    // Add lake outflow rivers (from overflowing lakes) - always major
    generate_lake_outflow_vertices(world, &mut vertices, river_color);

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
    let Some(hydrology) = &world.hydrology else {
        return;
    };
    let elevation = world.elevation.as_ref().unwrap();
    let tessellation = &world.tessellation;

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

/// Generate river mesh for "major rivers" mode as triangle strips.
/// Each segment is a quad (2 triangles) with width based on flow.
pub fn generate_river_mesh_major(world: &World) -> RiverMesh {
    let Some(hydrology) = &world.hydrology else {
        return RiverMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    };
    let elevation = world.elevation.as_ref().unwrap();
    let tessellation = &world.tessellation;

    // Resolution-independent thresholds
    let (_, outlet_threshold, branch_threshold) = river_thresholds(tessellation.num_cells());

    // Compute which cells are part of major rivers
    let is_major = hydrology.compute_major_river_cells(outlet_threshold, branch_threshold);

    // Find max flow for width normalization
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .copied()
        .fold(0.0f32, f32::max);

    // River color - muted blue
    let river_color = Vec3::new(0.15, 0.35, 0.60);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for cell_idx in 0..tessellation.num_cells() {
        if !is_major[cell_idx] {
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
            river_color,
            &mut vertices,
            &mut indices,
        );
    }

    // Add lake outflow rivers
    generate_lake_outflow_quads(world, max_flow, river_color, &mut vertices, &mut indices);

    RiverMesh { vertices, indices }
}

/// Generate river mesh for "all rivers" mode as triangle strips.
pub fn generate_river_mesh_all(world: &World) -> RiverMesh {
    let Some(hydrology) = &world.hydrology else {
        return RiverMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    };
    let elevation = world.elevation.as_ref().unwrap();
    let tessellation = &world.tessellation;

    // Resolution-independent threshold
    let (min_flow, _, _) = river_thresholds(tessellation.num_cells());

    // Find max flow for width normalization
    let max_flow = hydrology
        .flow_accumulation
        .iter()
        .copied()
        .fold(0.0f32, f32::max);

    // River color - muted blue
    let river_color = Vec3::new(0.15, 0.35, 0.60);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for cell_idx in 0..tessellation.num_cells() {
        let flow = hydrology.flow_accumulation[cell_idx];

        if flow < min_flow || hydrology.is_submerged(cell_idx) {
            continue;
        }

        let Some(downstream_idx) = hydrology.downstream(cell_idx) else {
            continue;
        };

        generate_river_segment_quad(
            tessellation,
            elevation,
            hydrology,
            cell_idx,
            downstream_idx,
            flow,
            max_flow,
            river_color,
            &mut vertices,
            &mut indices,
        );
    }

    // Add lake outflow rivers
    generate_lake_outflow_quads(world, max_flow, river_color, &mut vertices, &mut indices);

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
    let Some(hydrology) = &world.hydrology else {
        return;
    };
    let elevation = world.elevation.as_ref().unwrap();
    let tessellation = &world.tessellation;

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
    let plates = world.plates.as_ref().unwrap();
    let dynamics = world.dynamics.as_ref().unwrap();
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
    let continental_cells = plates
        .cell_plate
        .iter()
        .filter(|&&p| dynamics.plate_type(p as usize) == PlateType::Continental)
        .count();
    let continental_pct = 100.0 * continental_cells as f32 / num_cells as f32;

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

// =============================================================================
// Wind particle buffer generation
// =============================================================================

/// Generate a vertex buffer for wind particle trail lines.
/// Each trail segment is a line from prev_pos to pos with per-vertex color.
pub fn generate_wind_particle_buffer(
    device: &wgpu::Device,
    trails: &[(Vec3, Vec3, Vec3)], // (start, end, color)
) -> (wgpu::Buffer, u32) {
    let mut vertices = Vec::with_capacity(trails.len() * 2);

    for &(start, end, color) in trails {
        // Both vertices of the line share the same normal (radial) and color
        let normal = ((start + end) * 0.5).normalize();
        vertices.push(MeshVertex::new(start, normal, color));
        vertices.push(MeshVertex::new(end, normal, color));
    }

    let buffer = create_vertex_buffer(device, &vertices, "wind_particle_vertex");
    let count = vertices.len() as u32;

    (buffer, count)
}
