//! GPU wind particle system.
//!
//! Simulates particles flowing along the wind field on the sphere surface.
//! Uses adjacency-based cell tracking for efficient wind sampling.
//!
//! # Architecture
//!
//! The system consists of:
//! 1. **Compute shader** (`wind_particles.wgsl`): Updates particle positions each frame
//! 2. **Render shader** (`particle_render.wgsl`): Draws particles as line trails
//!
//! Particles track their current Voronoi cell and walk the adjacency graph to find
//! which cell they're in. This is O(1) per frame since particles move slowly.
//!
//! Trail length is based on wind magnitude (independent of frame time), while
//! movement is time-normalized for consistent behavior across frame rates.

use bytemuck::{Pod, Zeroable};
use encase::{ShaderType, UniformBuffer};
use glam::{UVec2, Vec3};
use rand::Rng;
use wgpu::util::DeviceExt;

use super::GpuContext;
use crate::world::{Atmosphere, Tessellation};

/// Maximum number of neighbors per cell (must match shader).
const MAX_NEIGHBORS: usize = 12;

/// Default number of particles.
pub const DEFAULT_NUM_PARTICLES: u32 = 100_000;

/// GPU particle representation (must match shader struct exactly).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuParticle {
    position: [f32; 3],
    cell: u32,
    trail_end: [f32; 3],
    age: f32,
}

/// Cell center for GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuCellCenter {
    position: [f32; 3],
    _padding: f32,
}

/// Wind vector for GPU.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuWindVector {
    velocity: [f32; 3],
    _padding: f32,
}

/// Adjacency offset/count for a cell.
/// Padded to 16 bytes for proper GPU struct alignment in storage buffers.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuAdjacencyOffset {
    offset: u32,
    count: u32,
    _padding: [u32; 2],
}

/// Simulation uniforms (compute shader).
/// Uses encase for automatic WGSL struct alignment.
#[derive(ShaderType)]
struct ComputeUniforms {
    dt: f32,
    speed_scale: f32,
    trail_scale: f32,
    time: f32,
    num_particles: u32,
    num_cells: u32,
    _pad0: UVec2,
    max_age: f32,
    _pad1: Vec3,
}

/// Render uniforms for particle shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ParticleRenderUniforms {
    max_age: f32,
    relief_scale: f32,
    upper_wind_height: f32,
    is_surface_wind: u32, // 1 = surface (use elevation), 0 = upper (fixed height)
}

/// GPU wind particle system.
pub struct WindParticleSystem {
    // Compute pipeline
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,

    // Render pipeline
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    render_uniform_buffer: wgpu::Buffer,
    render_bind_group_layout: wgpu::BindGroupLayout,

    // Shared buffers
    particle_buffer: wgpu::Buffer,
    compute_uniform_buffer: wgpu::Buffer,
    wind_buffer: wgpu::Buffer,

    // State
    num_particles: u32,
    num_cells: u32,
    time: f32,
    max_age: f32,
    is_surface_wind: bool,
}

impl WindParticleSystem {
    /// Create a new particle system.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `tessellation` - World tessellation (for cell centers and adjacency)
    /// * `atmosphere` - Atmosphere data (for wind vectors)
    /// * `num_particles` - Number of particles to simulate
    /// * `main_bind_group_layout` - The main uniform bind group layout (for camera/view uniforms)
    /// * `elevation_map` - Elevation map texture for terrain height sampling
    /// * `rng` - Random number generator for initial positions
    pub fn new<R: Rng>(
        ctx: &GpuContext,
        tessellation: &Tessellation,
        atmosphere: &Atmosphere,
        num_particles: u32,
        main_bind_group_layout: &wgpu::BindGroupLayout,
        elevation_map: &super::ElevationMap,
        rng: &mut R,
    ) -> Self {
        let num_cells = tessellation.num_cells() as u32;
        let max_age = 2.4; // ~120 frames at 50fps

        // Build cell centers buffer
        let cell_centers: Vec<GpuCellCenter> = (0..tessellation.num_cells())
            .map(|i| {
                let pos = tessellation.cell_center(i);
                GpuCellCenter {
                    position: [pos.x, pos.y, pos.z],
                    _padding: 0.0,
                }
            })
            .collect();

        // Initialize particles at random cell centers
        let particles: Vec<GpuParticle> = (0..num_particles)
            .map(|_| {
                let cell = rng.gen_range(0..num_cells);
                let pos = tessellation.cell_center(cell as usize);
                GpuParticle {
                    position: [pos.x, pos.y, pos.z],
                    cell,
                    trail_end: [pos.x, pos.y, pos.z],
                    age: rng.gen_range(0.0..max_age * 0.5),
                }
            })
            .collect();

        // Build wind buffer
        let wind_vectors: Vec<GpuWindVector> = atmosphere
            .wind
            .iter()
            .map(|v| GpuWindVector {
                velocity: [v.x, v.y, v.z],
                _padding: 0.0,
            })
            .collect();

        // Build adjacency data (flattened)
        let mut adjacency_offsets = Vec::with_capacity(tessellation.num_cells());
        let mut adjacency_data = Vec::new();

        for i in 0..tessellation.num_cells() {
            let neighbors = tessellation.neighbors(i);
            let offset = adjacency_data.len() as u32;
            let count = neighbors.len().min(MAX_NEIGHBORS) as u32;

            adjacency_offsets.push(GpuAdjacencyOffset {
                offset,
                count,
                _padding: [0; 2],
            });

            for &n in neighbors.iter().take(MAX_NEIGHBORS) {
                adjacency_data.push(n as u32);
            }
        }

        // Create buffers
        let particle_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particle_buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        });

        let cell_center_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cell_center_buffer"),
                    contents: bytemuck::cast_slice(&cell_centers),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let wind_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wind_buffer"),
            contents: bytemuck::cast_slice(&wind_vectors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let adjacency_offset_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adjacency_offset_buffer"),
                    contents: bytemuck::cast_slice(&adjacency_offsets),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let adjacency_data_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adjacency_data_buffer"),
                    contents: bytemuck::cast_slice(&adjacency_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let compute_uniforms = ComputeUniforms {
            dt: 1.0 / 60.0,
            speed_scale: 0.5,
            trail_scale: 0.03,
            time: 0.0,
            num_particles,
            num_cells,
            _pad0: UVec2::ZERO,
            max_age,
            _pad1: Vec3::ZERO,
        };

        let mut uniform_buffer = UniformBuffer::new(Vec::new());
        uniform_buffer.write(&compute_uniforms).unwrap();

        let compute_uniform_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("particle_compute_uniform_buffer"),
                    contents: uniform_buffer.as_ref(),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // Create compute pipeline
        let compute_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("wind_particles_compute_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/wind_particles.wgsl").into(),
                ),
            });

        let compute_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("particle_compute_bind_group_layout"),
                    entries: &[
                        // Particles (read/write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Cell centers (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Wind (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Adjacency offsets (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Adjacency data (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let compute_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_compute_bind_group"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_center_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wind_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: adjacency_offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adjacency_data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: compute_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("particle_compute_pipeline_layout"),
                    bind_group_layouts: &[&compute_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("particle_compute_pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &compute_shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Create render pipeline
        let render_shader_source = format!(
            "{}\n{}",
            include_str!("../shaders/cubemap.wgsl"),
            include_str!("../shaders/particle_render.wgsl"),
        );
        let render_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("particle_render_shader"),
                source: wgpu::ShaderSource::Wgsl(render_shader_source.into()),
            });

        // Render bind group layout (particles + render uniforms + elevation texture)
        let render_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("particle_render_bind_group_layout"),
                    entries: &[
                        // Particles (read)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Render uniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Elevation cubemap
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Elevation sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        // RELIEF_SCALE from constants - particles need same displacement as terrain
        const RELIEF_SCALE: f32 = 0.2;
        const UPPER_WIND_HEIGHT: f32 = 1.06; // Slightly above sea level

        let render_uniforms = ParticleRenderUniforms {
            max_age,
            relief_scale: RELIEF_SCALE,
            upper_wind_height: UPPER_WIND_HEIGHT,
            is_surface_wind: 1, // Default to surface wind
        };

        let render_uniform_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("particle_render_uniform_buffer"),
                    contents: bytemuck::cast_slice(&[render_uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let render_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_render_bind_group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: render_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(elevation_map.array_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(elevation_map.sampler()),
                },
            ],
        });

        let render_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("particle_render_pipeline_layout"),
                    bind_group_layouts: &[main_bind_group_layout, &render_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let render_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("particle_render_pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[], // No vertex buffers - using storage buffer directly
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &render_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        Self {
            compute_pipeline,
            compute_bind_group,
            render_pipeline,
            render_bind_group,
            render_uniform_buffer,
            render_bind_group_layout,
            particle_buffer,
            compute_uniform_buffer,
            wind_buffer,
            num_particles,
            num_cells,
            time: 0.0,
            max_age,
            is_surface_wind: true,
        }
    }

    /// Update the wind buffer with new wind data and set wind layer type.
    ///
    /// Used to switch between surface wind and upper wind visualization.
    pub fn set_wind(&mut self, ctx: &GpuContext, wind: &[glam::Vec3], is_surface: bool) {
        let wind_vectors: Vec<GpuWindVector> = wind
            .iter()
            .map(|v| GpuWindVector {
                velocity: [v.x, v.y, v.z],
                _padding: 0.0,
            })
            .collect();
        ctx.queue
            .write_buffer(&self.wind_buffer, 0, bytemuck::cast_slice(&wind_vectors));

        // Update surface/upper wind flag
        if self.is_surface_wind != is_surface {
            self.is_surface_wind = is_surface;

            // RELIEF_SCALE from constants
            const RELIEF_SCALE: f32 = 0.2;
            const UPPER_WIND_HEIGHT: f32 = 1.06;

            let render_uniforms = ParticleRenderUniforms {
                max_age: self.max_age,
                relief_scale: RELIEF_SCALE,
                upper_wind_height: UPPER_WIND_HEIGHT,
                is_surface_wind: if is_surface { 1 } else { 0 },
            };
            ctx.queue.write_buffer(
                &self.render_uniform_buffer,
                0,
                bytemuck::cast_slice(&[render_uniforms]),
            );
        }
    }

    /// Update particles for one frame (dispatch compute shader).
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `dt` - Frame time in seconds (for movement and aging)
    /// * `speed_scale` - How much wind moves particles per second
    /// * `trail_scale` - Trail length per unit wind speed (independent of dt)
    pub fn update(&mut self, ctx: &GpuContext, dt: f32, speed_scale: f32, trail_scale: f32) {
        self.time += dt;

        // Update uniforms
        let uniforms = ComputeUniforms {
            dt,
            speed_scale,
            trail_scale,
            time: self.time,
            num_particles: self.num_particles,
            num_cells: self.num_cells,
            _pad0: UVec2::ZERO,
            max_age: self.max_age,
            _pad1: Vec3::ZERO,
        };
        let mut uniform_buffer = UniformBuffer::new(Vec::new());
        uniform_buffer.write(&uniforms).unwrap();
        ctx.queue
            .write_buffer(&self.compute_uniform_buffer, 0, uniform_buffer.as_ref());

        // Dispatch compute shader
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("particle_compute_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle_compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);

            // Dispatch with 64 threads per workgroup
            let workgroups = (self.num_particles + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Get the render pipeline for drawing particles.
    pub fn render_pipeline(&self) -> &wgpu::RenderPipeline {
        &self.render_pipeline
    }

    /// Get the render bind group (particles + uniforms).
    pub fn render_bind_group(&self) -> &wgpu::BindGroup {
        &self.render_bind_group
    }

    /// Get the number of particles.
    pub fn num_particles(&self) -> u32 {
        self.num_particles
    }

    /// Render particles to a render pass.
    ///
    /// The main uniforms bind group must already be set at group 0.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(1, &self.render_bind_group, &[]);
        // Draw 2 vertices per particle (line from trail_end to current position)
        render_pass.draw(0..2, 0..self.num_particles);
    }
}
