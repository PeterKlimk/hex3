mod coloring;
pub mod export;
mod state;
mod view;
mod visualization;
pub mod world;

use std::path::PathBuf;
use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{WindowAttributes, WindowId},
};

pub use state::AppState;
pub use view::{RenderMode, ViewMode};

/// Configuration for the app from CLI arguments.
pub struct AppConfig {
    pub seed: Option<u64>,
    pub target_stage: u32,
    pub export_path: Option<PathBuf>,
    pub gpu_voronoi: bool,
}

pub struct App {
    pub state: Option<AppState>,
    pub config: AppConfig,
}

impl App {
    pub fn new(config: AppConfig) -> Self {
        Self {
            state: None,
            config,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("Hex3 - Spherical Voronoi")
                        .with_inner_size(PhysicalSize::new(1280, 720)),
                )
                .expect("Failed to create window"),
        );

        let seed = self.config.seed.unwrap_or_else(rand::random);
        let mut state = pollster::block_on(AppState::new(window, seed, self.config.gpu_voronoi));

        // Advance to target stage
        while state.world_data.current_stage() < self.config.target_stage {
            if !state.advance_stage() {
                break;
            }
        }

        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => state.resize(new_size),
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Tab) => {
                            state.view_mode = match state.view_mode {
                                ViewMode::Globe => ViewMode::Map,
                                ViewMode::Map => ViewMode::Globe,
                            };
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit1) => {
                            state.render_mode = RenderMode::Relief;
                            // Relief uses unified mesh, no color regeneration needed
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            state.render_mode = RenderMode::Terrain;
                            state.regenerate_colors();
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit3) => {
                            state.render_mode = RenderMode::Elevation;
                            state.regenerate_colors();
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit4) => {
                            state.render_mode = RenderMode::Plates;
                            state.regenerate_colors();
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit5) => {
                            if state.render_mode == RenderMode::Noise {
                                // Already in noise mode - cycle through layers
                                state.noise_layer = state.noise_layer.cycle();
                                println!("Noise layer: {}", state.noise_layer.name());
                            } else {
                                state.render_mode = RenderMode::Noise;
                            }
                            state.regenerate_colors();
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit6) => {
                            state.render_mode = RenderMode::Hydrology;
                            state.regenerate_colors();
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit7) => {
                            if state.render_mode == RenderMode::Features {
                                // Already in features mode - cycle through layers
                                state.feature_layer = state.feature_layer.cycle();
                                println!("Feature layer: {}", state.feature_layer.name());
                            } else {
                                state.render_mode = RenderMode::Features;
                            }
                            state.regenerate_colors();
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit8) => {
                            if state.render_mode == RenderMode::Climate {
                                // Already in climate mode - cycle through layers
                                state.climate_layer = state.climate_layer.cycle();
                                println!("Climate layer: {}", state.climate_layer.name());
                            } else {
                                state.render_mode = RenderMode::Climate;
                            }
                            state.regenerate_colors();
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            let new_seed: u64 = rand::random();
                            state.regenerate_world(new_seed);
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            state.show_edges = !state.show_edges;
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::KeyV) => {
                            state.river_mode = state.river_mode.cycle();
                            println!("Rivers: {}", state.river_mode.name());
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Space) => {
                            state.advance_stage();
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            if state.adjust_climate(0.05) {
                                state.window.request_redraw();
                            }
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            if state.adjust_climate(-0.05) {
                                state.window.request_redraw();
                            }
                        }
                        PhysicalKey::Code(KeyCode::KeyH) => {
                            state.hemisphere_lighting = !state.hemisphere_lighting;
                            println!(
                                "Hemisphere lighting: {}",
                                if state.hemisphere_lighting {
                                    "ON"
                                } else {
                                    "OFF"
                                }
                            );
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::KeyD) => {
                            // Export world data
                            let filename = format!(
                                "hex3_dump_{}_{}_{}.json.gz",
                                state.seed,
                                state.world_data.tessellation.num_cells(),
                                state.world_data.current_stage()
                            );
                            let path = std::path::PathBuf::from(&filename);
                            export::export_world(&state.world_data, state.seed, &path);
                        }
                        PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => {
                if state.view_mode == ViewMode::Globe {
                    match button_state {
                        ElementState::Pressed => state.camera_controller.on_mouse_press(),
                        ElementState::Released => state.camera_controller.on_mouse_release(),
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if state.view_mode == ViewMode::Globe
                    && state.camera_controller.on_mouse_move(
                        position.x as f32,
                        position.y as f32,
                        &mut state.camera,
                    )
                {
                    state.window.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if state.view_mode == ViewMode::Globe {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                    };
                    if state.camera_controller.on_scroll(scroll, &mut state.camera) {
                        state.window.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested => state.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
