use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Mat4, Vec3};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use wgpu::{
    include_wgsl, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, PrimitiveTopology, ShaderStages,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use hex3::geometry::{
    lloyd_relax, random_sphere_points_with_rng, tectonics::PlateType, MeshVertex, SphericalVoronoi,
    TectonicPlates, VoronoiMesh,
};
use hex3::render::{
    create_depth_texture, create_index_buffer, create_uniform_buffer, create_vertex_buffer,
    CameraController, GpuContext, OrbitCamera, PipelineBuilder, Uniforms,
};

const NUM_CELLS: usize = 20000;
const LLOYD_ITERATIONS: usize = 5;
const NUM_PLATES: usize = 14;

#[derive(Clone, Copy, PartialEq)]
enum ViewMode {
    Globe,
    Map,
}

#[derive(Clone, Copy, PartialEq)]
enum RenderMode {
    Elevation,
    Plates,
    Stress,
    Relief,
    Noise,
}

impl RenderMode {
    fn name(&self) -> &'static str {
        match self {
            RenderMode::Elevation => "Elevation",
            RenderMode::Plates => "Plates",
            RenderMode::Stress => "Stress",
            RenderMode::Relief => "Relief",
            RenderMode::Noise => "Noise",
        }
    }
}

struct App {
    state: Option<AppState>,
}

/// Per-render-mode vertex buffers
struct RenderModeBuffers {
    globe_vertex_buffer: wgpu::Buffer,
    map_vertex_buffer: wgpu::Buffer,
    // Edge line segments (non-indexed, pairs of vertices)
    globe_edge_vertex_buffer: wgpu::Buffer,
    num_globe_edge_vertices: u32,
}

/// All regeneratable world data (buffers that change when the world is regenerated)
struct WorldBuffers {
    elevation_buffers: RenderModeBuffers,
    plates_buffers: RenderModeBuffers,
    stress_buffers: RenderModeBuffers,
    relief_buffers: RenderModeBuffers,
    noise_buffers: RenderModeBuffers,
    map_edge_vertex_buffer: wgpu::Buffer,
    map_edge_index_buffer: wgpu::Buffer,
    globe_index_buffer: wgpu::Buffer,
    map_index_buffer: wgpu::Buffer,
    arrow_vertex_buffer: wgpu::Buffer,
    pole_marker_vertex_buffer: wgpu::Buffer,
    pole_marker_index_buffer: wgpu::Buffer,
    num_arrow_vertices: u32,
    num_pole_marker_indices: u32,
    num_globe_indices: u32,
    num_map_indices: u32,
    num_map_edge_indices: u32,
}

struct AppState {
    window: Arc<Window>,
    gpu: GpuContext,
    camera: OrbitCamera,
    camera_controller: CameraController,

    // View and render mode
    view_mode: ViewMode,
    render_mode: RenderMode,

    // Pipelines (static - don't change on regeneration)
    fill_pipeline: wgpu::RenderPipeline,
    map_fill_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    colored_line_pipeline: wgpu::RenderPipeline,

    // Uniform buffer and bind group (static)
    uniform_buffer: wgpu::Buffer,
    bind_group: BindGroup,

    // Depth texture (static - only changes on resize)
    depth_view: wgpu::TextureView,

    // World data (regenerated on R key)
    world: WorldBuffers,

    // Current world seed
    seed: u64,

    // Display options
    show_edges: bool,

    // FPS tracking
    frame_count: u32,
    fps_update_time: Instant,
    current_fps: f32,
}

impl App {
    fn new() -> Self {
        Self { state: None }
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

        let state = pollster::block_on(create_app_state(window));
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
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                state.gpu.resize(new_size);
                state.camera.set_aspect(state.gpu.aspect());
                let (_, depth_view) =
                    create_depth_texture(&state.gpu.device, new_size.width, new_size.height);
                state.depth_view = depth_view;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::Tab) => {
                            // Toggle view mode
                            state.view_mode = match state.view_mode {
                                ViewMode::Globe => ViewMode::Map,
                                ViewMode::Map => ViewMode::Globe,
                            };
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit1) => {
                            state.render_mode = RenderMode::Elevation;
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit2) => {
                            state.render_mode = RenderMode::Plates;
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit3) => {
                            state.render_mode = RenderMode::Stress;
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit4) => {
                            state.render_mode = RenderMode::Relief;
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Digit5) => {
                            state.render_mode = RenderMode::Noise;
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            // Regenerate world with new random seed
                            let new_seed: u64 = rand::random();
                            state.world = generate_world_buffers(&state.gpu.device, new_seed);
                            state.seed = new_seed;
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::KeyE) => {
                            // Toggle edge visibility
                            state.show_edges = !state.show_edges;
                            state.window.request_redraw();
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => {
                // Only allow camera control in globe mode
                if state.view_mode == ViewMode::Globe {
                    match button_state {
                        ElementState::Pressed => {
                            state.camera_controller.on_mouse_press();
                        }
                        ElementState::Released => {
                            state.camera_controller.on_mouse_release();
                        }
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
            WindowEvent::RedrawRequested => {
                render(state);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

/// Generate world buffers with a specific seed.
fn generate_world_buffers(device: &wgpu::Device, seed: u64) -> WorldBuffers {
    let total_start = Instant::now();
    println!("Generating world with seed: {}", seed);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    print!("Generating {} random points... ", NUM_CELLS);
    let start = Instant::now();
    let mut points = random_sphere_points_with_rng(NUM_CELLS, &mut rng);
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    print!("Lloyd relaxation ({} iterations)... ", LLOYD_ITERATIONS);
    let start = Instant::now();
    lloyd_relax(&mut points, LLOYD_ITERATIONS);
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    print!("Computing Voronoi... ");
    let start = Instant::now();
    let voronoi = SphericalVoronoi::compute(&points);
    println!(
        "{:.1}ms ({} cells)",
        start.elapsed().as_secs_f64() * 1000.0,
        voronoi.cells.len()
    );

    print!("Generating plates... ");
    let start = Instant::now();
    let plates = TectonicPlates::generate_with_rng(&voronoi, NUM_PLATES, &mut rng);
    println!(
        "{:.1}ms ({} plates)",
        start.elapsed().as_secs_f64() * 1000.0,
        NUM_PLATES
    );

    print!("Building meshes for render modes... ");
    let start = Instant::now();

    // Build mesh for each render mode
    let mesh_elevation =
        VoronoiMesh::from_voronoi_with_colors(&voronoi, |i| plates.cell_color_elevation(i));
    let mesh_plates =
        VoronoiMesh::from_voronoi_with_colors(&voronoi, |i| plates.cell_color_plate(i));
    let mesh_stress =
        VoronoiMesh::from_voronoi_with_colors(&voronoi, |i| plates.cell_color_stress(i));
    let mesh_relief = VoronoiMesh::from_voronoi_with_elevation(
        &voronoi,
        |i| plates.cell_color_elevation(i),
        |i| plates.cell_elevation[i],
    );
    let mesh_noise =
        VoronoiMesh::from_voronoi_with_colors(&voronoi, |i| plates.cell_color_noise(i));

    // Generate edge line segments for each mode
    let dark_color = Vec3::new(0.1, 0.1, 0.1);
    let edge_vertices_dark = VoronoiMesh::edge_lines_with_colors(&voronoi, |_, _| dark_color);

    // Build boundary edge color map for plates mode
    let boundary_edge_colors = plates.build_boundary_edge_colors(&voronoi);
    let edge_vertices_plates = VoronoiMesh::edge_lines_with_colors(&voronoi, |a, b| {
        let key = if a < b { (a, b) } else { (b, a) };
        boundary_edge_colors.get(&key).copied().unwrap_or(dark_color)
    });

    // Relief mode edges with elevation displacement
    let edge_vertices_relief = VoronoiMesh::edge_lines_with_elevation(
        &voronoi,
        |_, _| dark_color,
        |i| plates.cell_elevation[i],
    );
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Generate plate overlay geometry
    print!("Generating plate overlays... ");
    let start = Instant::now();
    let arrows = plates.generate_velocity_arrows(&voronoi);
    let pole_markers = plates.generate_pole_markers();

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

    // Create 2D map vertices
    print!("Map projection... ");
    let start = Instant::now();
    let (map_vertices_elevation, map_indices) = mesh_elevation.to_map_vertices();
    let (map_vertices_plates, _) = mesh_plates.to_map_vertices();
    let (map_vertices_stress, _) = mesh_stress.to_map_vertices();
    let (map_vertices_relief, _) = mesh_relief.to_map_vertices();
    let (map_vertices_noise, _) = mesh_noise.to_map_vertices();
    let (map_edge_vertices, map_edge_indices) = mesh_elevation.to_map_edge_vertices(&voronoi);
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Create buffers
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
            &edge_vertices_dark,
            "elevation_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_dark.len() as u32,
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
            &edge_vertices_dark,
            "stress_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_dark.len() as u32,
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
            &edge_vertices_dark,
            "noise_globe_edge_vertex",
        ),
        num_globe_edge_vertices: edge_vertices_dark.len() as u32,
    };

    // Print world statistics
    let num_cells = plates.cell_elevation.len();
    let water_count = plates.cell_elevation.iter().filter(|&&e| e < 0.0).count();
    let water_pct = 100.0 * water_count as f32 / num_cells as f32;
    let avg_elevation: f32 = plates.cell_elevation.iter().sum::<f32>() / num_cells as f32;
    let min_elevation = plates
        .cell_elevation
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_elevation = plates
        .cell_elevation
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let continental_cells = plates
        .cell_plate
        .iter()
        .filter(|&&p| plates.plate_types[p as usize] == PlateType::Continental)
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

    println!(
        "\nWorld generation: {:.1}ms",
        total_start.elapsed().as_secs_f64() * 1000.0
    );

    WorldBuffers {
        elevation_buffers,
        plates_buffers,
        stress_buffers,
        relief_buffers,
        noise_buffers,
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
            &mesh_elevation.indices,
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
        num_globe_indices: mesh_elevation.indices.len() as u32,
        num_map_indices: map_indices.len() as u32,
        num_map_edge_indices: map_edge_indices.len() as u32,
    }
}

async fn create_app_state(window: Arc<Window>) -> AppState {
    let total_start = Instant::now();

    print!("Initializing GPU... ");
    let start = Instant::now();
    let gpu = GpuContext::new(window.clone()).await;
    println!("{:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    // Generate initial seed from system entropy
    let seed: u64 = rand::random();

    // Generate world
    let world = generate_world_buffers(&gpu.device, seed);

    // Create camera
    let mut camera = OrbitCamera::new();
    camera.set_aspect(gpu.aspect());
    let camera_controller = CameraController::new();

    // Create shaders
    let fill_shader = gpu
        .device
        .create_shader_module(include_wgsl!("shaders/sphere.wgsl"));
    let edge_shader = gpu
        .device
        .create_shader_module(include_wgsl!("shaders/edge.wgsl"));
    let colored_line_shader = gpu
        .device
        .create_shader_module(include_wgsl!("shaders/colored_line.wgsl"));

    // Create uniform buffer and bind group
    let uniforms = Uniforms::new(
        camera.view_projection(),
        camera.eye_position(),
        Vec3::new(0.5, 1.0, 0.3).normalize(),
    );
    let uniform_buffer = create_uniform_buffer(&gpu.device, &uniforms, "uniforms");

    let bind_group_layout = gpu
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("uniform_bind_group_layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
        label: Some("uniform_bind_group"),
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    // Create pipelines
    let fill_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
        .shader(&fill_shader)
        .vertex_layout(MeshVertex::desc())
        .bind_group_layout(&bind_group_layout)
        .label("fill_pipeline")
        .build();

    let map_fill_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
        .shader(&fill_shader)
        .vertex_layout(MeshVertex::desc())
        .bind_group_layout(&bind_group_layout)
        .cull_mode(None)
        .label("map_fill_pipeline")
        .build();

    let edge_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
        .shader(&edge_shader)
        .vertex_layout(MeshVertex::desc())
        .bind_group_layout(&bind_group_layout)
        .topology(PrimitiveTopology::LineList)
        .cull_mode(None)
        .label("edge_pipeline")
        .build();

    let colored_line_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
        .shader(&colored_line_shader)
        .vertex_layout(MeshVertex::desc())
        .bind_group_layout(&bind_group_layout)
        .topology(PrimitiveTopology::LineList)
        .cull_mode(None)
        .label("colored_line_pipeline")
        .build();

    // Create depth texture
    let (_, depth_view) = create_depth_texture(&gpu.device, gpu.size.width, gpu.size.height);

    println!(
        "Total init: {:.1}ms",
        total_start.elapsed().as_secs_f64() * 1000.0
    );
    println!("Ready! Drag to rotate, scroll to zoom.");
    println!("  Tab: toggle map view");
    println!("  1-5: Elevation/Plates/Stress/Relief/Noise modes");
    println!("  E: toggle edges");
    println!("  R: regenerate world");

    AppState {
        window,
        gpu,
        camera,
        camera_controller,
        view_mode: ViewMode::Globe,
        render_mode: RenderMode::Elevation,
        fill_pipeline,
        map_fill_pipeline,
        edge_pipeline,
        colored_line_pipeline,
        uniform_buffer,
        bind_group,
        depth_view,
        world,
        seed,
        show_edges: true,
        frame_count: 0,
        fps_update_time: Instant::now(),
        current_fps: 0.0,
    }
}

fn render(state: &mut AppState) {
    // Choose projection based on view mode
    let (view_proj, camera_pos) = match state.view_mode {
        ViewMode::Globe => (state.camera.view_projection(), state.camera.eye_position()),
        ViewMode::Map => {
            // Orthographic projection for 2D map
            // Use slightly wider bounds to show wrapped cells at edges
            let aspect = state.gpu.aspect();
            let proj = if aspect > 2.2 {
                // Very wide screen - fit height with padding
                Mat4::orthographic_rh(-aspect, aspect, -1.1, 1.1, -1.0, 1.0)
            } else {
                // Normal/tall screen - fit width with padding for wrapped cells
                let half_height = 1.1 / aspect;
                Mat4::orthographic_rh(-1.1, 1.1, -half_height, half_height, -1.0, 1.0)
            };
            (proj, Vec3::new(0.0, 0.0, 1.0))
        }
    };

    // Light direction - from above-front for map, normal for globe
    let light_dir = match state.view_mode {
        ViewMode::Globe => Vec3::new(0.5, 1.0, 0.3).normalize(),
        ViewMode::Map => Vec3::new(0.0, 0.0, 1.0), // Straight on
    };

    let uniforms = Uniforms::new(view_proj, camera_pos, light_dir);
    state
        .gpu
        .queue
        .write_buffer(&state.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

    // Choose render mode buffers
    let mode_buffers = match state.render_mode {
        RenderMode::Elevation => &state.world.elevation_buffers,
        RenderMode::Plates => &state.world.plates_buffers,
        RenderMode::Stress => &state.world.stress_buffers,
        RenderMode::Relief => &state.world.relief_buffers,
        RenderMode::Noise => &state.world.noise_buffers,
    };

    // Choose buffers and pipeline based on view mode
    let (vertex_buffer, index_buffer, num_indices, fill_pipeline) = match state.view_mode {
        ViewMode::Globe => (
            &mode_buffers.globe_vertex_buffer,
            &state.world.globe_index_buffer,
            state.world.num_globe_indices,
            &state.fill_pipeline,
        ),
        ViewMode::Map => (
            &mode_buffers.map_vertex_buffer,
            &state.world.map_index_buffer,
            state.world.num_map_indices,
            &state.map_fill_pipeline, // No backface culling for map
        ),
    };

    // Get surface texture
    let output = match state.gpu.surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Lost) => {
            state.gpu.resize(state.gpu.size);
            return;
        }
        Err(wgpu::SurfaceError::OutOfMemory) => {
            panic!("Out of GPU memory");
        }
        Err(e) => {
            eprintln!("Surface error: {:?}", e);
            return;
        }
    };
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_encoder"),
        });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.15,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &state.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Draw filled cells
        render_pass.set_pipeline(fill_pipeline);
        render_pass.set_bind_group(0, &state.bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..num_indices, 0, 0..1);

        // Draw edges (different approach for globe vs map)
        if state.show_edges {
            match state.view_mode {
                ViewMode::Globe => {
                    // Globe: per-mode colored edges (non-indexed, vertex colors)
                    render_pass.set_pipeline(&state.colored_line_pipeline);
                    render_pass.set_vertex_buffer(0, mode_buffers.globe_edge_vertex_buffer.slice(..));
                    render_pass.draw(0..mode_buffers.num_globe_edge_vertices, 0..1);
                }
                ViewMode::Map => {
                    // Map: shared dark edges (indexed)
                    render_pass.set_pipeline(&state.edge_pipeline);
                    render_pass.set_vertex_buffer(0, state.world.map_edge_vertex_buffer.slice(..));
                    render_pass.set_index_buffer(
                        state.world.map_edge_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..state.world.num_map_edge_indices, 0, 0..1);
                }
            }
        }

        // Draw plate overlays in Plates mode (globe view only)
        if state.render_mode == RenderMode::Plates && state.view_mode == ViewMode::Globe {
            // Draw velocity arrows as colored lines
            if state.world.num_arrow_vertices > 0 {
                render_pass.set_pipeline(&state.colored_line_pipeline);
                render_pass.set_vertex_buffer(0, state.world.arrow_vertex_buffer.slice(..));
                render_pass.draw(0..state.world.num_arrow_vertices, 0..1);
            }

            // Draw pole markers as triangles
            if state.world.num_pole_marker_indices > 0 {
                render_pass.set_pipeline(&state.fill_pipeline);
                render_pass.set_vertex_buffer(0, state.world.pole_marker_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    state.world.pole_marker_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..state.world.num_pole_marker_indices, 0, 0..1);
            }
        }
    }

    state.gpu.queue.submit(std::iter::once(encoder.finish()));
    output.present();

    // Update FPS counter
    state.frame_count += 1;
    let now = Instant::now();
    let elapsed = now.duration_since(state.fps_update_time);
    if elapsed >= Duration::from_secs(1) {
        state.current_fps = state.frame_count as f32 / elapsed.as_secs_f32();
        state.frame_count = 0;
        state.fps_update_time = now;

        let view = match state.view_mode {
            ViewMode::Globe => "Globe",
            ViewMode::Map => "Map",
        };
        state.window.set_title(&format!(
            "Hex3 - {} | {} | {:.0} FPS",
            view,
            state.render_mode.name(),
            state.current_fps
        ));
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new();
    event_loop
        .run_app(&mut app)
        .expect("Failed to run application");
}
