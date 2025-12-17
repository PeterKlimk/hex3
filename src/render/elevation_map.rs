//! Elevation cubemap texture for continuous terrain height sampling.
//!
//! Renders terrain elevation to a cubemap texture that can be sampled
//! from any shader to get smooth, interpolated terrain height at any point on the sphere.
//! Cubemap avoids the singularities and seams of equirectangular projection.

use glam::{Mat4, Vec3};

use super::{CubemapTexture, GpuContext};

/// Resolution of each cubemap face.
pub const ELEVATION_MAP_FACE_SIZE: u32 = 512;

/// Elevation cubemap for continuous terrain height sampling.
pub struct ElevationMap {
    pub cubemap: CubemapTexture,
    /// Render pipeline for filling the texture
    render_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for the render pipeline
    bind_group_layout: wgpu::BindGroupLayout,
    /// Uniform buffer for view-projection matrix
    uniform_buffer: wgpu::Buffer,
    /// Depth texture for proper occlusion (shared for all faces)
    depth_texture: wgpu::Texture,
    /// Depth texture view
    depth_view: wgpu::TextureView,
}

/// View matrices for each cubemap face (looking outward from center)
fn cubemap_view_matrices() -> [Mat4; 6] {
    [
        // +X: forward +X, up -Y
        Mat4::look_at_rh(Vec3::ZERO, Vec3::X, -Vec3::Y),
        // -X: forward -X, up -Y
        Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_X, -Vec3::Y),
        // +Y: forward +Y, up +Z
        Mat4::look_at_rh(Vec3::ZERO, Vec3::Y, Vec3::Z),
        // -Y: forward -Y, up -Z
        Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Y, -Vec3::Z),
        // +Z: forward +Z, up -Y
        Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, -Vec3::Y),
        // -Z: forward -Z, up -Y
        Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, -Vec3::Y),
    ]
}

/// 90-degree FOV projection for cubemap faces
fn cubemap_projection() -> Mat4 {
    Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 10.0)
}

impl ElevationMap {
    /// Create a new elevation cubemap.
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::util::DeviceExt;

        let cubemap = CubemapTexture::new(
            device,
            "elevation_cubemap",
            ELEVATION_MAP_FACE_SIZE,
            wgpu::TextureFormat::R16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            },
        );

        // Depth texture for proper occlusion (reused for each face)
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("elevation_cubemap_depth"),
            size: wgpu::Extent3d {
                width: ELEVATION_MAP_FACE_SIZE,
                height: ELEVATION_MAP_FACE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create shader for rendering elevation to cubemap
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elevation_cubemap_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/elevation_map.wgsl").into()),
        });

        // Bind group layout - view_proj uniform
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("elevation_cubemap_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("elevation_cubemap_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Vertex layout: position (vec3) + elevation (f32)
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("elevation_cubemap_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create uniform buffer for view-projection matrix
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("elevation_cubemap_uniform_buffer"),
            contents: bytemuck::cast_slice(&[Mat4::IDENTITY]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            cubemap,
            render_pipeline,
            bind_group_layout,
            uniform_buffer,
            depth_texture,
            depth_view,
        }
    }

    /// Render elevation data to the cubemap.
    ///
    /// Takes vertex buffer with position + elevation and index buffer.
    /// Renders to all 6 faces with appropriate view matrices.
    pub fn render(
        &self,
        ctx: &GpuContext,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        num_indices: u32,
    ) {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elevation_cubemap_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        });

        let proj = cubemap_projection();
        let views = cubemap_view_matrices();

        // Render to each face - submit separately to ensure uniform buffer updates are applied
        for (face_idx, (face_view, view_matrix)) in
            self.cubemap.face_views.iter().zip(views.iter()).enumerate()
        {
            // Update view-projection matrix for this face
            let view_proj = proj * *view_matrix;
            ctx.queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[view_proj]),
            );

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("elevation_cubemap_encoder_{}", face_idx)),
                });

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(&format!("elevation_cubemap_face_{}_pass", face_idx)),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: face_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0, // Default to sea level
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..num_indices, 0, 0..1);
            }

            ctx.queue.submit(std::iter::once(encoder.finish()));
        }
    }

    pub fn array_view(&self) -> &wgpu::TextureView {
        &self.cubemap.array_view
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.cubemap.sampler
    }
}

/// Vertex format for elevation map rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ElevationVertex {
    pub position: [f32; 3],
    pub elevation: f32,
}
