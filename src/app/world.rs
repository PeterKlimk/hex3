use std::time::Instant;

use glam::Vec3;

use hex3::geometry::{Material, MeshVertex, SurfaceVertex, UnifiedMesh, UnifiedVertex, VoronoiMesh};
use hex3::render::{create_index_buffer, create_vertex_buffer};
use hex3::world::{PlateType, World};

use super::coloring::{
    cell_color_elevation, cell_color_hydrology, cell_color_noise, cell_color_plate,
    cell_color_stress, cell_color_terrain, cell_material,
};
use super::view::{NoiseLayer, RenderMode};
use super::visualization::{build_boundary_edge_colors, generate_pole_markers, generate_velocity_arrows};

pub const NUM_CELLS: usize = 40000;
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

/// Per-render-mode vertex buffers.
pub struct RenderModeBuffers {
    pub globe_vertex_buffer: wgpu::Buffer,
    pub map_vertex_buffer: wgpu::Buffer,
    pub globe_edge_vertex_buffer: wgpu::Buffer,
    pub num_globe_edge_vertices: u32,
}

/// All regeneratable world data (buffers that change when the world is regenerated).
pub struct WorldBuffers {
    pub mode_buffers: [RenderModeBuffers; RenderMode::COUNT],
    pub map_edge_vertex_buffer: wgpu::Buffer,
    pub map_edge_index_buffer: wgpu::Buffer,
    pub globe_index_buffer: wgpu::Buffer,
    pub map_index_buffer: wgpu::Buffer,
    pub arrow_vertex_buffer: wgpu::Buffer,
    pub pole_marker_vertex_buffer: wgpu::Buffer,
    pub pole_marker_index_buffer: wgpu::Buffer,
    pub num_arrow_vertices: u32,
    pub num_pole_marker_indices: u32,
    pub num_globe_indices: u32,
    pub num_map_indices: u32,
    pub num_map_edge_indices: u32,
    // Legacy river data (line-based) - kept for non-relief modes
    pub river_all_vertex_buffer: wgpu::Buffer,
    pub river_major_vertex_buffer: wgpu::Buffer,
    pub num_river_all_vertices: u32,
    pub num_river_major_vertices: u32,
    // Triangle-based river mesh (Phase 2) - uses UnifiedVertex with Material::River
    pub river_mesh_all_vertex_buffer: wgpu::Buffer,
    pub river_mesh_all_index_buffer: wgpu::Buffer,
    pub river_mesh_major_vertex_buffer: wgpu::Buffer,
    pub river_mesh_major_index_buffer: wgpu::Buffer,
    pub num_river_mesh_all_indices: u32,
    pub num_river_mesh_major_indices: u32,
    // Unified mesh buffers (material-aware lighting) for relief mode
    pub unified_globe_vertex_buffer: wgpu::Buffer,
    pub unified_map_vertex_buffer: wgpu::Buffer,
    pub unified_globe_index_buffer: wgpu::Buffer,
    pub unified_map_index_buffer: wgpu::Buffer,
    pub num_unified_globe_indices: u32,
    pub num_unified_map_indices: u32,
}

impl WorldBuffers {
    pub fn for_mode(&self, mode: RenderMode) -> &RenderModeBuffers {
        &self.mode_buffers[mode.idx()]
    }

    /// Get the river buffer and vertex count for the given river mode.
    pub fn river_buffer(&self, mode: super::view::RiverMode) -> Option<(&wgpu::Buffer, u32)> {
        match mode {
            super::view::RiverMode::Off => None,
            super::view::RiverMode::Major => Some((&self.river_major_vertex_buffer, self.num_river_major_vertices)),
            super::view::RiverMode::All => Some((&self.river_all_vertex_buffer, self.num_river_all_vertices)),
        }
    }
}

/// Generate a new world (Stage 1: Lithosphere).
pub fn create_world(seed: u64) -> World {
    let total_start = Instant::now();
    println!("Generating world with seed: {}", seed);

    print!(
        "Generating tessellation ({} cells, {} Lloyd iterations)... ",
        NUM_CELLS, LLOYD_ITERATIONS
    );
    let start = Instant::now();
    let mut world = World::new(seed, NUM_CELLS, LLOYD_ITERATIONS);
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    print!("Generating plates... ");
    let start = Instant::now();
    world.generate_plates(NUM_PLATES);
    println!(
        "{:.1}ms ({} plates)",
        start.elapsed().as_secs_f64() * 1000.0,
        NUM_PLATES
    );

    print!("Generating dynamics... ");
    let start = Instant::now();
    world.generate_dynamics();
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    print!("Generating stress... ");
    let start = Instant::now();
    world.generate_stress();
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    print!("Generating elevation... ");
    let start = Instant::now();
    world.generate_elevation();
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    print_world_stats(&world);

    println!(
        "\nStage 1 (Lithosphere): {:.1}ms",
        total_start.elapsed().as_secs_f64() * 1000.0
    );

    world
}

/// Advance world to Stage 2 (Hydrology).
pub fn advance_to_stage_2(world: &mut World) {
    let start = Instant::now();
    print!("Generating hydrology... ");
    world.generate_hydrology();
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Print hydrology stats
    if let Some(hydrology) = &world.hydrology {
        let num_cells = world.num_cells();

        let ocean_cells = (0..num_cells).filter(|&i| hydrology.is_ocean(i)).count();
        let lake_cells = (0..num_cells).filter(|&i| hydrology.is_lake_water(i)).count();
        let dry_basin_cells = (0..num_cells).filter(|&i| hydrology.is_dry_basin(i)).count();
        let land_cells = (0..num_cells).filter(|&i| !hydrology.is_submerged(i)).count();
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

        println!(
            "  Ocean: {} cells, Land: {} cells, Lakes: {} cells ({:.1}% of non-ocean)",
            ocean_cells, land_cells, lake_cells, lake_pct
        );
        println!(
            "  Basins: {} total, {} dry",
            hydrology.basins.len(),
            dry_basin_cells
        );
        println!("  Drainage coverage: {} cells", cells_with_drainage);
        println!(
            "  Rivers (flow >= {:.0}): {} cells, max flow: {:.0}",
            river_min_flow,
            river_cells.len(),
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

/// Generate GPU buffers from a World.
pub fn generate_world_buffers(
    device: &wgpu::Device,
    world: &World,
    noise_layer: NoiseLayer,
) -> WorldBuffers {
    let voronoi = &world.tessellation.voronoi;
    let elevation = world.elevation.as_ref().unwrap();

    print!("Building meshes for render modes... ");
    let start = Instant::now();

    // Terrain mode: elevation coloring + lakes (default view)
    let mesh_terrain =
        VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_terrain(world, i));
    let mesh_elevation =
        VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_elevation(world, i));
    let mesh_plates =
        VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_plate(world, i));
    let mesh_stress =
        VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_stress(world, i));
    // Relief mode uses terrain coloring (with lakes) + elevation displacement
    // Water surfaces are flat (elevation 0 for ocean, water_level for lakes)
    // Micro noise is added back for visual relief (it's excluded from simulation elevation)
    let mesh_relief = VoronoiMesh::from_voronoi_with_elevation(
        voronoi,
        |i| cell_color_terrain(world, i),
        |i| {
            if let Some(hydrology) = &world.hydrology {
                if hydrology.is_ocean(i) {
                    return 0.0;
                }
                if hydrology.is_lake_water(i) {
                    return hydrology.basin(i).map(|b| b.water_level).unwrap_or(0.0);
                }
            }
            // Simulation elevation only
            elevation.values[i]
        },
    );

    // Unified mesh with material-aware lighting for relief mode
    // Water surfaces are flat (elevation 0 for ocean, water_level for lakes)
    // Shader adds procedural micro noise for land/river materials
    let unified_mesh = UnifiedMesh::from_voronoi_with_elevation(
        voronoi,
        |i| cell_color_terrain(world, i),
        |i| cell_material(world, i),
        |i| {
            if let Some(hydrology) = &world.hydrology {
                if hydrology.is_ocean(i) {
                    return 0.0; // Ocean surface at sea level
                }
                if hydrology.is_lake_water(i) {
                    // Lake surface at water level
                    return hydrology.basin(i).map(|b| b.water_level).unwrap_or(0.0);
                }
            }
            // Simulation elevation only - shader adds micro noise
            elevation.values[i]
        },
    );

    let mesh_noise =
        VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_noise(world, i, noise_layer));
    let mesh_hydrology =
        VoronoiMesh::from_voronoi_with_colors(voronoi, |i| cell_color_hydrology(world, i));

    // Subtle gray for cell borders
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
            elevation.values[i]
        },
        |i| cell_material(world, i),
    );
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    print!("Generating plate overlays... ");
    let start = Instant::now();
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

    println!(
        "{:.1}ms ({} arrows, {} pole markers, {} boundary edges colored)",
        start.elapsed().as_secs_f64() * 1000.0,
        arrows.len() / 3,
        pole_markers.len() / 3,
        boundary_edge_colors.len()
    );

    // Generate river vertices using SurfaceVertex (legacy line-based)
    let river_all_vertices = generate_river_vertices_all(world);
    let river_major_vertices = generate_river_vertices_major(world);

    // Generate triangle-based river meshes (Phase 2)
    let river_mesh_all = generate_river_mesh_all(world);
    let river_mesh_major = generate_river_mesh_major(world);

    if !river_all_vertices.is_empty() {
        println!(
            "  River segments: {} line, {} triangles (major: {} line, {} triangles)",
            river_all_vertices.len() / 2,
            river_mesh_all.indices.len() / 3,
            river_major_vertices.len() / 2,
            river_mesh_major.indices.len() / 3,
        );
    }

    print!("Map projection... ");
    let start = Instant::now();
    let (map_vertices_terrain, map_indices) = mesh_terrain.to_map_vertices();
    let (map_vertices_elevation, _) = mesh_elevation.to_map_vertices();
    let (map_vertices_plates, _) = mesh_plates.to_map_vertices();
    let (map_vertices_stress, _) = mesh_stress.to_map_vertices();
    let (map_vertices_relief, _) = mesh_relief.to_map_vertices();
    let (map_vertices_noise, _) = mesh_noise.to_map_vertices();
    let (map_vertices_hydrology, _) = mesh_hydrology.to_map_vertices();
    let (map_edge_vertices, map_edge_indices) = mesh_terrain.to_map_edge_vertices(voronoi);
    // Unified mesh map projection
    let (unified_map_vertices, unified_map_indices) = unified_mesh.to_map_vertices();
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    let terrain_buffers = RenderModeBuffers {
        globe_vertex_buffer: create_vertex_buffer(
            device,
            &mesh_terrain.vertices,
            "terrain_globe_vertex",
        ),
        map_vertex_buffer: create_vertex_buffer(
            device,
            &map_vertices_terrain,
            "terrain_map_vertex",
        ),
        globe_edge_vertex_buffer: create_vertex_buffer(
            device,
            &edge_vertices_default,
            "terrain_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_default.len() as u32,
    };

    let elevation_buffers = RenderModeBuffers {
        globe_vertex_buffer: create_vertex_buffer(
            device,
            &mesh_elevation.vertices,
            "elevation_globe_vertex",
        ),
        map_vertex_buffer: create_vertex_buffer(
            device,
            &map_vertices_elevation,
            "elevation_map_vertex",
        ),
        globe_edge_vertex_buffer: create_vertex_buffer(
            device,
            &edge_vertices_default,
            "elevation_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_default.len() as u32,
    };

    let plates_buffers = RenderModeBuffers {
        globe_vertex_buffer: create_vertex_buffer(
            device,
            &mesh_plates.vertices,
            "plates_globe_vertex",
        ),
        map_vertex_buffer: create_vertex_buffer(device, &map_vertices_plates, "plates_map_vertex"),
        globe_edge_vertex_buffer: create_vertex_buffer(
            device,
            &edge_vertices_plates,
            "plates_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_plates.len() as u32,
    };

    let stress_buffers = RenderModeBuffers {
        globe_vertex_buffer: create_vertex_buffer(
            device,
            &mesh_stress.vertices,
            "stress_globe_vertex",
        ),
        map_vertex_buffer: create_vertex_buffer(device, &map_vertices_stress, "stress_map_vertex"),
        globe_edge_vertex_buffer: create_vertex_buffer(
            device,
            &edge_vertices_default,
            "stress_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_default.len() as u32,
    };

    let relief_buffers = RenderModeBuffers {
        globe_vertex_buffer: create_vertex_buffer(
            device,
            &mesh_relief.vertices,
            "relief_globe_vertex",
        ),
        map_vertex_buffer: create_vertex_buffer(device, &map_vertices_relief, "relief_map_vertex"),
        globe_edge_vertex_buffer: create_vertex_buffer(
            device,
            &edge_vertices_relief,
            "relief_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_relief.len() as u32,
    };

    let noise_buffers = RenderModeBuffers {
        globe_vertex_buffer: create_vertex_buffer(
            device,
            &mesh_noise.vertices,
            "noise_globe_vertex",
        ),
        map_vertex_buffer: create_vertex_buffer(device, &map_vertices_noise, "noise_map_vertex"),
        globe_edge_vertex_buffer: create_vertex_buffer(
            device,
            &edge_vertices_default,
            "noise_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_default.len() as u32,
    };

    let hydrology_buffers = RenderModeBuffers {
        globe_vertex_buffer: create_vertex_buffer(
            device,
            &mesh_hydrology.vertices,
            "hydrology_globe_vertex",
        ),
        map_vertex_buffer: create_vertex_buffer(
            device,
            &map_vertices_hydrology,
            "hydrology_map_vertex",
        ),
        globe_edge_vertex_buffer: create_vertex_buffer(
            device,
            &edge_vertices_default,
            "hydrology_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_default.len() as u32,
    };

    WorldBuffers {
        // Order matches RenderMode: Relief, Terrain, Elevation, Plates, Stress, Noise, Hydrology
        mode_buffers: [
            relief_buffers,
            terrain_buffers,
            elevation_buffers,
            plates_buffers,
            stress_buffers,
            noise_buffers,
            hydrology_buffers,
        ],
        map_edge_vertex_buffer: create_vertex_buffer(
            device,
            &map_edge_vertices,
            "map_edge_vertex_buffer",
        ),
        map_edge_index_buffer: create_index_buffer(
            device,
            &map_edge_indices,
            "map_edge_index_buffer",
        ),
        globe_index_buffer: create_index_buffer(
            device,
            &mesh_terrain.indices,
            "globe_index_buffer",
        ),
        map_index_buffer: create_index_buffer(device, &map_indices, "map_index_buffer"),
        arrow_vertex_buffer: create_vertex_buffer(device, &arrow_vertices, "arrow_vertex_buffer"),
        pole_marker_vertex_buffer: create_vertex_buffer(
            device,
            &pole_marker_vertices,
            "pole_marker_vertex_buffer",
        ),
        pole_marker_index_buffer: create_index_buffer(
            device,
            &pole_marker_indices,
            "pole_marker_index_buffer",
        ),
        num_arrow_vertices: arrow_vertices.len() as u32,
        num_pole_marker_indices: pole_marker_indices.len() as u32,
        num_globe_indices: mesh_terrain.indices.len() as u32,
        num_map_indices: map_indices.len() as u32,
        num_map_edge_indices: map_edge_indices.len() as u32,
        river_all_vertex_buffer: create_vertex_buffer(device, &river_all_vertices, "river_all_vertex_buffer"),
        river_major_vertex_buffer: create_vertex_buffer(device, &river_major_vertices, "river_major_vertex_buffer"),
        num_river_all_vertices: river_all_vertices.len() as u32,
        num_river_major_vertices: river_major_vertices.len() as u32,
        // Triangle-based river mesh buffers (Phase 2)
        river_mesh_all_vertex_buffer: create_vertex_buffer(device, &river_mesh_all.vertices, "river_mesh_all_vertex"),
        river_mesh_all_index_buffer: create_index_buffer(device, &river_mesh_all.indices, "river_mesh_all_index"),
        river_mesh_major_vertex_buffer: create_vertex_buffer(device, &river_mesh_major.vertices, "river_mesh_major_vertex"),
        river_mesh_major_index_buffer: create_index_buffer(device, &river_mesh_major.indices, "river_mesh_major_index"),
        num_river_mesh_all_indices: river_mesh_all.indices.len() as u32,
        num_river_mesh_major_indices: river_mesh_major.indices.len() as u32,
        // Unified mesh buffers (material-aware lighting)
        unified_globe_vertex_buffer: create_vertex_buffer(device, &unified_mesh.vertices, "unified_globe_vertex"),
        unified_map_vertex_buffer: create_vertex_buffer(device, &unified_map_vertices, "unified_map_vertex"),
        unified_globe_index_buffer: create_index_buffer(device, &unified_mesh.indices, "unified_globe_index"),
        unified_map_index_buffer: create_index_buffer(device, &unified_map_indices, "unified_map_index"),
        num_unified_globe_indices: unified_mesh.indices.len() as u32,
        num_unified_map_indices: unified_map_indices.len() as u32,
    }
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

        vertices.push(SurfaceVertex::new(start_pos, start_elev, river_color, alpha));
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

        vertices.push(SurfaceVertex::new(start_pos, start_elev, river_color, alpha));
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
fn generate_lake_outflow_vertices(world: &World, vertices: &mut Vec<SurfaceVertex>, river_color: Vec3) {
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

            let (start_pos, end_pos, start_elev, end_elev) =
                river_segment_geometry(tessellation, elevation, hydrology, cell_idx, downstream_idx);

            vertices.push(SurfaceVertex::new(start_pos, start_elev, river_color, outflow_alpha));
            vertices.push(SurfaceVertex::new(end_pos, end_elev, river_color, outflow_alpha));
        }

        // Add final segment to water (if path ends at land cell adjacent to water)
        if let Some(&last_cell) = path.last() {
            if let Some(downstream_idx) = hydrology.downstream(last_cell) {
                if hydrology.is_submerged(downstream_idx) {
                    let (start_pos, end_pos, start_elev, end_elev) =
                        river_segment_geometry(tessellation, elevation, hydrology, last_cell, downstream_idx);

                    vertices.push(SurfaceVertex::new(start_pos, start_elev, river_color, outflow_alpha));
                    vertices.push(SurfaceVertex::new(end_pos, end_elev, river_color, outflow_alpha));
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
    vertices.push(UnifiedVertex::new(p0, n1, color, start_elev, Material::River));
    vertices.push(UnifiedVertex::new(p1, n1, color, start_elev, Material::River));
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

    println!("\n=== World Statistics ===");
    println!("  Cells: {}", num_cells);
    println!("  Water: {:.1}%", water_pct);
    println!("  Continental crust: {:.1}%", continental_pct);
    println!(
        "  Elevation: avg {:.3}, min {:.3}, max {:.3}",
        avg_elevation, min_elevation, max_elevation
    );
}
