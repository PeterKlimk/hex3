use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Mat4, Vec3};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use hex3::render::{
    CameraController, FillPipelineKind, GpuContext, IndexedDraw, LayerUniforms, LineDraw,
    OrbitCamera, RenderScene, Renderer, SurfaceLineDraw, Uniforms,
};
use hex3::world::{World, DEFAULT_CLIMATE_RATIO};

use super::view::{FeatureLayer, NoiseLayer, RenderMode, RiverMode, ViewMode};
use super::world::{advance_to_stage_2, create_world, generate_world_buffers, WorldBuffers};

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

    pub show_edges: bool,
    pub river_mode: RiverMode,
    pub noise_layer: NoiseLayer,
    pub feature_layer: FeatureLayer,

    /// Rendering effect toggle
    pub hemisphere_lighting: bool,

    pub frame_count: u32,
    pub fps_update_time: Instant,
    pub current_fps: f32,
}

impl AppState {
    pub async fn new(window: Arc<Window>) -> Self {
        let total_start = Instant::now();

        print!("Initializing GPU... ");
        let start = Instant::now();
        let gpu = GpuContext::new(window.clone()).await;
        println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

        let seed: u64 = rand::random();
        let world_data = create_world(seed);
        let noise_layer = NoiseLayer::Combined;
        let feature_layer = FeatureLayer::default();
        let world_buffers = generate_world_buffers(&gpu.device, &world_data, noise_layer, feature_layer);

        let mut camera = OrbitCamera::new();
        camera.set_aspect(gpu.aspect());
        let camera_controller = CameraController::new();

        let initial_uniforms = Uniforms::new(
            camera.view_projection(),
            camera.eye_position(),
            Vec3::new(0.5, 1.0, 0.3).normalize(),
        );
        let renderer = Renderer::new(&gpu, &initial_uniforms);

        println!(
            "Total init: {:.1}ms",
            total_start.elapsed().as_secs_f64() * 1000.0
        );
        println!("Ready! Drag to rotate, scroll to zoom.");
        println!("  Tab: toggle map view");
        println!("  1-8: Relief/Terrain/Elevation/Plates/Stress/Noise/Hydrology/Features");
        println!("  E: toggle edges | V: cycle rivers (Off/Major/All)");
        println!("  H: toggle hemisphere lighting");
        println!("  R: regenerate | Space: advance stage");
        println!("  Up/Down: adjust climate (wetter/drier) [Stage 2]");

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
            show_edges: false,
            river_mode: RiverMode::Major,
            noise_layer: NoiseLayer::Combined,
            feature_layer: FeatureLayer::default(),
            hemisphere_lighting: true,
            frame_count: 0,
            fps_update_time: Instant::now(),
            current_fps: 0.0,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.gpu.resize(new_size);
        self.camera.set_aspect(self.gpu.aspect());
        self.renderer
            .resize(&self.gpu.device, self.gpu.size.width, self.gpu.size.height);
    }

    pub fn regenerate_world(&mut self, seed: u64) {
        self.world_data = create_world(seed);
        self.world_buffers =
            generate_world_buffers(&self.gpu.device, &self.world_data, self.noise_layer, self.feature_layer);
        self.seed = seed;
    }

    /// Advance to the next simulation stage.
    /// Returns true if advanced, false if already at max stage.
    pub fn advance_stage(&mut self) -> bool {
        let current = self.world_data.current_stage();
        match current {
            1 => {
                advance_to_stage_2(&mut self.world_data);
                self.world_buffers =
                    generate_world_buffers(&self.gpu.device, &self.world_data, self.noise_layer, self.feature_layer);
                println!("Advanced to Stage 2: Hydrosphere");
                true
            }
            _ => {
                println!("Already at max stage ({})", current);
                false
            }
        }
    }

    /// Adjust climate ratio (precipitation/evaporation) by delta.
    /// Only works if hydrology is generated (Stage 2+).
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
        let overflowing = hydrology.basins.iter().filter(|b| b.is_overflowing()).count();
        let with_water = hydrology.basins.iter().filter(|b| b.has_water()).count();

        // Now regenerate buffers (needs immutable borrow of world_data)
        self.world_buffers =
            generate_world_buffers(&self.gpu.device, &self.world_data, self.noise_layer, self.feature_layer);

        println!(
            "Climate ratio: {:.2} | Lakes: {} with water, {} overflowing",
            new_ratio, with_water, overflowing
        );
        true
    }

    /// Get current climate ratio (or default if hydrology not generated).
    pub fn climate_ratio(&self) -> f32 {
        self.world_data
            .hydrology
            .as_ref()
            .map(|h| h.climate_ratio())
            .unwrap_or(DEFAULT_CLIMATE_RATIO)
    }

    pub fn render(&mut self) {
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

        // Enable relief displacement for relief mode
        let relief_enabled = self.render_mode.is_relief();

        let uniforms = Uniforms::new(view_proj, camera_pos, light_dir)
            .with_relief(relief_enabled)
            .with_hemisphere_lighting(self.hemisphere_lighting);

        let mode_buffers = self.world_buffers.for_mode(self.render_mode);

        // Determine which pipeline to use based on render mode
        let use_unified = self.render_mode == RenderMode::Relief;
        let use_layered_noise = self.render_mode == RenderMode::Noise;
        let use_layered_features = self.render_mode == RenderMode::Features;

        // Set layer uniforms if using layered pipeline
        if use_layered_noise {
            self.renderer.set_layer_uniforms(
                &self.gpu,
                &LayerUniforms::noise(self.noise_layer.idx() as u32),
            );
        } else if use_layered_features {
            self.renderer.set_layer_uniforms(
                &self.gpu,
                &LayerUniforms::features(self.feature_layer.idx() as u32),
            );
        }

        let (fill_pipeline, fill) = match (self.view_mode, use_unified, use_layered_noise, use_layered_features) {
            // Unified pipeline for Relief mode
            (ViewMode::Globe, true, _, _) => (
                FillPipelineKind::UnifiedGlobe,
                IndexedDraw {
                    vertex_buffer: &self.world_buffers.unified_globe_vertex_buffer,
                    index_buffer: &self.world_buffers.unified_globe_index_buffer,
                    index_count: self.world_buffers.num_unified_globe_indices,
                },
            ),
            (ViewMode::Map, true, _, _) => (
                FillPipelineKind::UnifiedMap,
                IndexedDraw {
                    vertex_buffer: &self.world_buffers.unified_map_vertex_buffer,
                    index_buffer: &self.world_buffers.unified_map_index_buffer,
                    index_count: self.world_buffers.num_unified_map_indices,
                },
            ),
            // Layered pipeline for Noise mode
            (ViewMode::Globe, _, true, _) => (
                FillPipelineKind::LayeredGlobe,
                IndexedDraw {
                    vertex_buffer: &self.world_buffers.noise_globe_vertex_buffer,
                    index_buffer: &self.world_buffers.noise_globe_index_buffer,
                    index_count: self.world_buffers.num_noise_globe_indices,
                },
            ),
            (ViewMode::Map, _, true, _) => (
                FillPipelineKind::LayeredMap,
                IndexedDraw {
                    vertex_buffer: &self.world_buffers.noise_map_vertex_buffer,
                    index_buffer: &self.world_buffers.noise_map_index_buffer,
                    index_count: self.world_buffers.num_noise_map_indices,
                },
            ),
            // Layered pipeline for Features mode
            (ViewMode::Globe, _, _, true) => (
                FillPipelineKind::LayeredGlobe,
                IndexedDraw {
                    vertex_buffer: &self.world_buffers.features_globe_vertex_buffer,
                    index_buffer: &self.world_buffers.features_globe_index_buffer,
                    index_count: self.world_buffers.num_features_globe_indices,
                },
            ),
            (ViewMode::Map, _, _, true) => (
                FillPipelineKind::LayeredMap,
                IndexedDraw {
                    vertex_buffer: &self.world_buffers.features_map_vertex_buffer,
                    index_buffer: &self.world_buffers.features_map_index_buffer,
                    index_count: self.world_buffers.num_features_map_indices,
                },
            ),
            // Legacy pipeline for other modes
            (ViewMode::Globe, _, _, _) => (
                FillPipelineKind::Globe,
                IndexedDraw {
                    vertex_buffer: &mode_buffers.globe_vertex_buffer,
                    index_buffer: &self.world_buffers.globe_index_buffer,
                    index_count: self.world_buffers.num_globe_indices,
                },
            ),
            (ViewMode::Map, _, _, _) => (
                FillPipelineKind::Map,
                IndexedDraw {
                    vertex_buffer: &mode_buffers.map_vertex_buffer,
                    index_buffer: &self.world_buffers.map_index_buffer,
                    index_count: self.world_buffers.num_map_indices,
                },
            ),
        };

        let edges = if self.show_edges {
            Some(match self.view_mode {
                ViewMode::Globe => hex3::render::EdgeDraw::GlobeColored(LineDraw {
                    vertex_buffer: &mode_buffers.globe_edge_vertex_buffer,
                    vertex_count: mode_buffers.num_globe_edge_vertices,
                }),
                ViewMode::Map => hex3::render::EdgeDraw::MapIndexed {
                    vertex_buffer: &self.world_buffers.map_edge_vertex_buffer,
                    index_buffer: &self.world_buffers.map_edge_index_buffer,
                    index_count: self.world_buffers.num_map_edge_indices,
                },
            })
        } else {
            None
        };

        let (arrows, pole_markers) = if self.render_mode == RenderMode::Plates
            && self.view_mode == ViewMode::Globe
        {
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
                    super::view::RiverMode::Off => None,
                    super::view::RiverMode::Major => {
                        (self.world_buffers.num_river_mesh_major_indices > 0).then_some(IndexedDraw {
                            vertex_buffer: &self.world_buffers.river_mesh_major_vertex_buffer,
                            index_buffer: &self.world_buffers.river_mesh_major_index_buffer,
                            index_count: self.world_buffers.num_river_mesh_major_indices,
                        })
                    }
                    super::view::RiverMode::All => {
                        (self.world_buffers.num_river_mesh_all_indices > 0).then_some(IndexedDraw {
                            vertex_buffer: &self.world_buffers.river_mesh_all_vertex_buffer,
                            index_buffer: &self.world_buffers.river_mesh_all_index_buffer,
                            index_count: self.world_buffers.num_river_mesh_all_indices,
                        })
                    }
                };
                (None, mesh)
            } else {
                // Other modes: use line-based rivers
                let lines = self.world_buffers
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
