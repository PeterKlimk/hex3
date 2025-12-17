use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Mat4, Vec3};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use winit::dpi::PhysicalSize;
use winit::window::Window;

use hex3::render::{
    CameraController, ElevationMap, FillPipelineKind, GpuContext, IndexedDraw, LineDraw,
    OrbitCamera, RenderScene, Renderer, SurfaceLineDraw, Uniforms, WindParticleSystem,
    DEFAULT_NUM_PARTICLES,
};
use hex3::world::World;

use super::view::{ClimateLayer, FeatureLayer, NoiseLayer, RenderMode, RiverMode, ViewMode};
use super::world::{
    advance_to_stage_2, advance_to_stage_3, create_world_with_options, generate_colored_mesh,
    generate_elevation_mesh_buffers, generate_world_buffers, WorldBuffers,
};

pub struct AppState {
    pub window: Arc<Window>,
    pub gpu: GpuContext,
    pub renderer: Renderer,
    pub camera: OrbitCamera,
    pub camera_controller: CameraController,

    pub view_mode: ViewMode,
    pub render_mode: RenderMode,

    pub world_data: World,
    pub world_buffers: WorldBuffers,
    pub seed: u64,
    pub gpu_voronoi: bool,

    pub show_edges: bool,
    pub river_mode: RiverMode,
    pub noise_layer: NoiseLayer,
    pub feature_layer: FeatureLayer,
    pub climate_layer: ClimateLayer,

    /// Rendering effect toggle
    pub hemisphere_lighting: bool,

    /// GPU wind particle system (for Climate/Wind visualization).
    pub gpu_particles: Option<WindParticleSystem>,
    /// Elevation map texture for terrain height sampling in shaders.
    pub elevation_map: ElevationMap,
    /// RNG for particle initialization.
    pub rng: ChaCha8Rng,

    pub frame_count: u32,
    pub fps_update_time: Instant,
    pub current_fps: f32,

    last_frame_time: Instant,
    wind_particle_speed_scale: f32,
    wind_particle_trail_scale: f32,
}

impl AppState {
    pub async fn new(window: Arc<Window>, seed: u64, gpu_voronoi: bool) -> Self {
        let total_start = Instant::now();

        print!("Initializing GPU... ");
        let start = Instant::now();
        let gpu = GpuContext::new(window.clone()).await;
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

        let world_data = create_world_with_options(seed, gpu_voronoi);
        let world_buffers = generate_world_buffers(&gpu.device, &world_data);

        let mut camera = OrbitCamera::new();
        camera.set_aspect(gpu.aspect());
        let camera_controller = CameraController::new();

        let initial_uniforms = Uniforms::new(
            camera.view_projection(),
            camera.eye_position(),
            Vec3::new(0.5, 1.0, 0.3).normalize(),
        );
        let renderer = Renderer::new(&gpu, &initial_uniforms);

        // Create elevation map and render initial elevation texture
        let elevation_map = ElevationMap::new(&gpu.device);
        let (elev_vertex_buf, elev_index_buf, elev_num_indices) =
            generate_elevation_mesh_buffers(&gpu.device, &world_data);
        elevation_map.render(&gpu, &elev_vertex_buf, &elev_index_buf, elev_num_indices);

        println!(
            "Total init: {:.1}ms (seed: {})",
            total_start.elapsed().as_secs_f64() * 1000.0,
            seed
        );
        println!("Ready! Drag to rotate, scroll to zoom.");
        println!("  Tab: toggle map view");
        println!("  1-8: Relief/Terrain/Elevation/Plates/Noise/Hydrology/Features/Climate");
        println!("  E: toggle edges | V: cycle rivers (Off/Major/All)");
        println!("  H: toggle hemisphere lighting | D: export data");
        println!("  R: regenerate | Space: advance stage");
        println!("  Up/Down: adjust climate (wetter/drier) [Stage 3]");

        Self {
            window,
            gpu,
            renderer,
            camera,
            camera_controller,
            view_mode: ViewMode::Globe,
            render_mode: RenderMode::Relief,
            world_data,
            world_buffers,
            seed,
            gpu_voronoi,
            show_edges: false,
            river_mode: RiverMode::Major,
            noise_layer: NoiseLayer::Combined,
            feature_layer: FeatureLayer::default(),
            climate_layer: ClimateLayer::default(),
            hemisphere_lighting: true,
            gpu_particles: None,
            elevation_map,
            rng: ChaCha8Rng::seed_from_u64(seed),
            frame_count: 0,
            fps_update_time: Instant::now(),
            current_fps: 0.0,
            last_frame_time: Instant::now(),
            wind_particle_speed_scale: 0.5,
            wind_particle_trail_scale: 0.03,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.gpu.resize(new_size);
        self.camera.set_aspect(self.gpu.aspect());
        self.renderer
            .resize(&self.gpu.device, self.gpu.size.width, self.gpu.size.height);
    }

    pub fn regenerate_world(&mut self, seed: u64) {
        self.world_data = create_world_with_options(seed, self.gpu_voronoi);
        self.world_buffers = generate_world_buffers(&self.gpu.device, &self.world_data);
        self.seed = seed;
        self.rng = ChaCha8Rng::seed_from_u64(seed);
        self.gpu_particles = None; // Will be recreated on stage advance

        // Re-render elevation map for the new terrain
        self.update_elevation_map();
    }

    /// Re-render the elevation map texture with current terrain data.
    /// Call this whenever elevation data changes (new world, stage advance, etc.)
    fn update_elevation_map(&self) {
        let (elev_vertex_buf, elev_index_buf, elev_num_indices) =
            generate_elevation_mesh_buffers(&self.gpu.device, &self.world_data);
        self.elevation_map.render(
            &self.gpu,
            &elev_vertex_buf,
            &elev_index_buf,
            elev_num_indices,
        );
    }

    /// Regenerate the colored mesh for the current mode/layer settings.
    /// Fast operation (~5-10ms) called on mode or layer switch.
    pub fn regenerate_colors(&mut self) {
        let (vertex_buffer, index_buffer, num_indices) = generate_colored_mesh(
            &self.gpu.device,
            &self.world_data,
            self.render_mode,
            self.noise_layer,
            self.feature_layer,
            self.climate_layer,
        );
        self.world_buffers.colored_vertex_buffer = vertex_buffer;
        self.world_buffers.colored_index_buffer = index_buffer;
        self.world_buffers.num_colored_indices = num_indices;
    }

    /// Advance to the next simulation stage.
    /// Returns true if advanced, false if already at max stage.
    pub fn advance_stage(&mut self) -> bool {
        let current = self.world_data.current_stage();
        match current {
            1 => {
                advance_to_stage_2(&mut self.world_data);
                self.world_buffers = generate_world_buffers(&self.gpu.device, &self.world_data);
                // Create GPU wind particles now that atmosphere exists
                if let Some(atmosphere) = &self.world_data.atmosphere {
                    self.gpu_particles = Some(WindParticleSystem::new(
                        &self.gpu,
                        &self.world_data.tessellation,
                        atmosphere,
                        DEFAULT_NUM_PARTICLES,
                        self.renderer.bind_group_layout(),
                        &self.elevation_map,
                        &mut self.rng,
                    ));
                }
                println!("Advanced to Stage 2: Atmosphere");
                true
            }
            2 => {
                advance_to_stage_3(&mut self.world_data);
                self.world_buffers = generate_world_buffers(&self.gpu.device, &self.world_data);
                println!("Advanced to Stage 3: Hydrosphere");
                true
            }
            _ => {
                println!("Already at max stage ({})", current);
                false
            }
        }
    }

    /// Adjust climate ratio (precipitation/evaporation) by delta.
    /// Only works if hydrology is generated (Stage 3+).
    /// Returns true if climate was changed.
    pub fn adjust_climate(&mut self, delta: f32) -> bool {
        // Check if hydrology exists and get current ratio
        let Some(hydrology) = &mut self.world_data.hydrology else {
            return false;
        };

        let old_ratio = hydrology.climate_ratio();
        let new_ratio = (old_ratio + delta).clamp(0.0, 2.0);

        if (new_ratio - old_ratio).abs() <= 0.001 {
            return false;
        }

        hydrology.set_climate_ratio(&self.world_data.tessellation, new_ratio);

        // Count stats before releasing the borrow
        let overflowing = hydrology
            .basins
            .iter()
            .filter(|b| b.is_overflowing())
            .count();
        let with_water = hydrology.basins.iter().filter(|b| b.has_water()).count();

        // Now regenerate buffers (needs immutable borrow of world_data)
        self.world_buffers = generate_world_buffers(&self.gpu.device, &self.world_data);

        println!(
            "Climate ratio: {:.2} | Lakes: {} with water, {} overflowing",
            new_ratio, with_water, overflowing
        );
        true
    }

    pub fn render(&mut self) {
        // Real frame time (used to tick fixed-rate simulations independent of render FPS).
        let now = Instant::now();
        let frame_dt = (now - self.last_frame_time).as_secs_f32().clamp(0.0, 0.1);
        self.last_frame_time = now;

        let (view_proj, camera_pos) = match self.view_mode {
            ViewMode::Globe => (self.camera.view_projection(), self.camera.eye_position()),
            ViewMode::Map => {
                let aspect = self.gpu.aspect();
                let proj = if aspect > 2.2 {
                    Mat4::orthographic_rh(-aspect, aspect, -1.1, 1.1, -1.0, 1.0)
                } else {
                    let half_height = 1.1 / aspect;
                    Mat4::orthographic_rh(-1.1, 1.1, -half_height, half_height, -1.0, 1.0)
                };
                (proj, Vec3::new(0.0, 0.0, 1.0))
            }
        };

        let light_dir = match self.view_mode {
            ViewMode::Globe => Vec3::new(0.5, 1.0, 0.3).normalize(),
            ViewMode::Map => Vec3::new(0.0, 0.0, 1.0),
        };

        // Wind layers use relief terrain so particles match the 3D surface
        let is_wind_layer = self.render_mode == RenderMode::Climate
            && matches!(
                self.climate_layer,
                ClimateLayer::Wind | ClimateLayer::UpperWind
            );

        // Enable relief displacement for relief mode and wind layers
        let relief_enabled = self.render_mode.is_relief() || is_wind_layer;

        // Enable map mode for shader-based projection
        let map_mode_enabled = self.view_mode == ViewMode::Map;

        let uniforms = Uniforms::new(view_proj, camera_pos, light_dir)
            .with_relief(relief_enabled)
            .with_hemisphere_lighting(self.hemisphere_lighting)
            .with_map_mode(map_mode_enabled);

        // Select pipeline and buffers based on render mode
        // Wind layers use unified (relief) mesh so particles align with terrain
        let use_unified = self.render_mode == RenderMode::Relief || is_wind_layer;

        let fill_pipeline = match (self.view_mode, use_unified) {
            (ViewMode::Globe, true) => FillPipelineKind::UnifiedGlobe,
            (ViewMode::Map, true) => FillPipelineKind::UnifiedMap,
            (ViewMode::Globe, false) => FillPipelineKind::Globe,
            (ViewMode::Map, false) => FillPipelineKind::Map,
        };

        // Select vertex/index buffers: unified for Relief, colored for all other modes
        let fill = if use_unified {
            IndexedDraw {
                vertex_buffer: &self.world_buffers.unified_vertex_buffer,
                index_buffer: &self.world_buffers.unified_index_buffer,
                index_count: self.world_buffers.num_unified_indices,
            }
        } else {
            IndexedDraw {
                vertex_buffer: &self.world_buffers.colored_vertex_buffer,
                index_buffer: &self.world_buffers.colored_index_buffer,
                index_count: self.world_buffers.num_colored_indices,
            }
        };

        // Select edge buffer based on mode
        let edges = if self.show_edges {
            let (buffer, count) = if use_unified {
                (
                    &self.world_buffers.relief_edge_vertex_buffer,
                    self.world_buffers.num_relief_edge_vertices,
                )
            } else if self.render_mode == RenderMode::Plates {
                (
                    &self.world_buffers.edge_vertex_buffer_plates,
                    self.world_buffers.num_edge_vertices_plates,
                )
            } else {
                (
                    &self.world_buffers.edge_vertex_buffer,
                    self.world_buffers.num_edge_vertices,
                )
            };
            Some(hex3::render::EdgeDraw::GlobeColored(LineDraw {
                vertex_buffer: buffer,
                vertex_count: count,
            }))
        } else {
            None
        };

        let (arrows, pole_markers) =
            if self.render_mode == RenderMode::Plates && self.view_mode == ViewMode::Globe {
                let arrows = (self.world_buffers.num_arrow_vertices > 0).then_some(LineDraw {
                    vertex_buffer: &self.world_buffers.arrow_vertex_buffer,
                    vertex_count: self.world_buffers.num_arrow_vertices,
                });

                let pole_markers =
                    (self.world_buffers.num_pole_marker_indices > 0).then_some(IndexedDraw {
                        vertex_buffer: &self.world_buffers.pole_marker_vertex_buffer,
                        index_buffer: &self.world_buffers.pole_marker_index_buffer,
                        index_count: self.world_buffers.num_pole_marker_indices,
                    });

                (arrows, pole_markers)
            } else {
                (None, None)
            };

        // Rivers: Use triangle mesh for Relief mode, line-based for other modes
        let (rivers, river_mesh) = if self.view_mode == ViewMode::Globe {
            if use_unified {
                // Relief mode: use triangle-based rivers
                let mesh = match self.river_mode {
                    RiverMode::Off => None,
                    RiverMode::Major => (self.world_buffers.num_river_mesh_major_indices > 0)
                        .then_some(IndexedDraw {
                            vertex_buffer: &self.world_buffers.river_mesh_major_vertex_buffer,
                            index_buffer: &self.world_buffers.river_mesh_major_index_buffer,
                            index_count: self.world_buffers.num_river_mesh_major_indices,
                        }),
                    RiverMode::All => (self.world_buffers.num_river_mesh_all_indices > 0)
                        .then_some(IndexedDraw {
                            vertex_buffer: &self.world_buffers.river_mesh_all_vertex_buffer,
                            index_buffer: &self.world_buffers.river_mesh_all_index_buffer,
                            index_count: self.world_buffers.num_river_mesh_all_indices,
                        }),
                };
                (None, mesh)
            } else {
                // Other modes: use line-based rivers
                let lines = self
                    .world_buffers
                    .river_buffer(self.river_mode)
                    .filter(|(_, count)| *count > 0)
                    .map(|(buffer, count)| SurfaceLineDraw {
                        vertex_buffer: buffer,
                        vertex_count: count,
                    });
                (lines, None)
            }
        } else {
            (None, None)
        };

        // GPU wind particles: update and render when in Climate/Wind mode on Globe view
        let gpu_particles = if self.view_mode == ViewMode::Globe
            && self.render_mode == RenderMode::Climate
            && matches!(
                self.climate_layer,
                ClimateLayer::Wind | ClimateLayer::UpperWind
            ) {
            if let Some(particles) = &mut self.gpu_particles {
                // Simple per-frame update: movement is time-normalized, trail length is fixed
                particles.update(
                    &self.gpu,
                    frame_dt,
                    self.wind_particle_speed_scale,
                    self.wind_particle_trail_scale,
                );
                Some(particles as &WindParticleSystem)
            } else {
                None
            }
        } else {
            None
        };

        self.renderer.render(
            &mut self.gpu,
            &uniforms,
            RenderScene {
                fill_pipeline,
                fill,
                edges,
                arrows,
                pole_markers,
                rivers,
                river_mesh,
                wind_particles: None, // Legacy CPU particles no longer used
                gpu_particles,
            },
        );

        self.update_fps();
    }

    fn update_fps(&mut self) {
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.fps_update_time);
        if elapsed < Duration::from_secs(1) {
            return;
        }

        self.current_fps = self.frame_count as f32 / elapsed.as_secs_f32();
        self.frame_count = 0;
        self.fps_update_time = now;

        let view = match self.view_mode {
            ViewMode::Globe => "Globe",
            ViewMode::Map => "Map",
        };
        let stage = self.world_data.current_stage();
        self.window.set_title(&format!(
            "Hex3 - {} | {} | Stage {} | {:.0} FPS",
            view,
            self.render_mode.name(),
            stage,
            self.current_fps
        ));
    }
}
