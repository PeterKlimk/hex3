use wgpu::{
    include_wgsl, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, Device, PrimitiveTopology,
    RenderPipeline, ShaderStages,
};

use super::{
    create_depth_texture, create_uniform_buffer, GpuContext, LayerUniforms, PipelineBuilder,
    Uniforms,
};
use crate::geometry::{LayeredVertex, MeshVertex, SurfaceVertex, UnifiedVertex};

/// Draw command for surface features (rivers, roads) using SurfaceVertex format.
pub struct SurfaceLineDraw<'a> {
    pub vertex_buffer: &'a Buffer,
    pub vertex_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillPipelineKind {
    /// Legacy globe pipeline (MeshVertex)
    Globe,
    /// Legacy map pipeline (MeshVertex)
    Map,
    /// Unified globe pipeline with material-aware lighting (UnifiedVertex)
    UnifiedGlobe,
    /// Unified map pipeline with material-aware lighting (UnifiedVertex)
    UnifiedMap,
    /// Layered globe pipeline for noise/features (LayeredVertex)
    LayeredGlobe,
    /// Layered map pipeline for noise/features (LayeredVertex)
    LayeredMap,
}

pub struct IndexedDraw<'a> {
    pub vertex_buffer: &'a Buffer,
    pub index_buffer: &'a Buffer,
    pub index_count: u32,
}

pub struct LineDraw<'a> {
    pub vertex_buffer: &'a Buffer,
    pub vertex_count: u32,
}

pub enum EdgeDraw<'a> {
    GlobeColored(LineDraw<'a>),
    MapIndexed {
        vertex_buffer: &'a Buffer,
        index_buffer: &'a Buffer,
        index_count: u32,
    },
}

pub struct RenderScene<'a> {
    pub fill_pipeline: FillPipelineKind,
    pub fill: IndexedDraw<'a>,
    pub edges: Option<EdgeDraw<'a>>,
    pub arrows: Option<LineDraw<'a>>,
    pub pole_markers: Option<IndexedDraw<'a>>,
    /// Legacy line-based rivers (SurfaceVertex)
    pub rivers: Option<SurfaceLineDraw<'a>>,
    /// Triangle-based rivers (UnifiedVertex) - uses unified pipeline
    pub river_mesh: Option<IndexedDraw<'a>>,
    /// Wind particle trails (MeshVertex lines)
    pub wind_particles: Option<LineDraw<'a>>,
}

pub struct Renderer {
    fill_pipeline: RenderPipeline,
    map_fill_pipeline: RenderPipeline,
    unified_fill_pipeline: RenderPipeline,
    map_unified_fill_pipeline: RenderPipeline,
    layered_fill_pipeline: RenderPipeline,
    map_layered_fill_pipeline: RenderPipeline,
    map_edge_pipeline: RenderPipeline,
    colored_line_pipeline: RenderPipeline,
    surface_line_pipeline: RenderPipeline,
    uniform_buffer: Buffer,
    layer_uniform_buffer: Buffer,
    bind_group: BindGroup,
    layered_bind_group: BindGroup,
    depth_view: wgpu::TextureView,
}

impl Renderer {
    pub fn new(gpu: &GpuContext, initial_uniforms: &Uniforms) -> Self {
        let fill_shader = gpu
            .device
            .create_shader_module(include_wgsl!("../shaders/sphere.wgsl"));
        let unified_shader = gpu
            .device
            .create_shader_module(include_wgsl!("../shaders/unified.wgsl"));
        let layered_shader = gpu
            .device
            .create_shader_module(include_wgsl!("../shaders/layered.wgsl"));
        let edge_shader = gpu
            .device
            .create_shader_module(include_wgsl!("../shaders/edge.wgsl"));
        let colored_line_shader = gpu
            .device
            .create_shader_module(include_wgsl!("../shaders/colored_line.wgsl"));
        let surface_line_shader = gpu
            .device
            .create_shader_module(include_wgsl!("../shaders/surface_line.wgsl"));

        let uniform_buffer = create_uniform_buffer(&gpu.device, initial_uniforms, "uniforms");
        let layer_uniform_buffer =
            create_uniform_buffer(&gpu.device, &LayerUniforms::noise(0), "layer_uniforms");

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

        // Layered bind group layout: main uniforms + layer uniforms
        let layered_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("layered_bind_group_layout"),
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform_bind_group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let layered_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("layered_bind_group"),
            layout: &layered_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: layer_uniform_buffer.as_entire_binding(),
                },
            ],
        });

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
            .depth_write(false)
            .label("map_fill_pipeline")
            .build();

        // Unified pipelines with material-aware lighting
        let unified_fill_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
            .shader(&unified_shader)
            .vertex_layout(UnifiedVertex::desc())
            .bind_group_layout(&bind_group_layout)
            .alpha_blend()
            .label("unified_fill_pipeline")
            .build();

        let map_unified_fill_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
            .shader(&unified_shader)
            .vertex_layout(UnifiedVertex::desc())
            .bind_group_layout(&bind_group_layout)
            .cull_mode(None)
            .depth_write(false)
            .alpha_blend()
            .label("map_unified_fill_pipeline")
            .build();

        // Layered pipelines for noise/features visualization
        let layered_fill_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
            .shader(&layered_shader)
            .vertex_layout(LayeredVertex::desc())
            .bind_group_layout(&layered_bind_group_layout)
            .label("layered_fill_pipeline")
            .build();

        let map_layered_fill_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
            .shader(&layered_shader)
            .vertex_layout(LayeredVertex::desc())
            .bind_group_layout(&layered_bind_group_layout)
            .cull_mode(None)
            .depth_write(false)
            .label("map_layered_fill_pipeline")
            .build();

        let map_edge_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
            .shader(&edge_shader)
            .vertex_layout(MeshVertex::desc())
            .bind_group_layout(&bind_group_layout)
            .topology(PrimitiveTopology::LineList)
            .cull_mode(None)
            .depth_write(false)
            .alpha_blend()
            .label("map_edge_pipeline")
            .build();

        let colored_line_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
            .shader(&colored_line_shader)
            .vertex_layout(MeshVertex::desc())
            .bind_group_layout(&bind_group_layout)
            .topology(PrimitiveTopology::LineList)
            .cull_mode(None)
            .alpha_blend()
            .label("colored_line_pipeline")
            .build();

        let surface_line_pipeline = PipelineBuilder::new(&gpu.device, gpu.format)
            .shader(&surface_line_shader)
            .vertex_layout(SurfaceVertex::desc())
            .bind_group_layout(&bind_group_layout)
            .topology(PrimitiveTopology::LineList)
            .cull_mode(None)
            .alpha_blend()
            .label("surface_line_pipeline")
            .build();

        let (_, depth_view) = create_depth_texture(&gpu.device, gpu.size.width, gpu.size.height);

        Self {
            fill_pipeline,
            map_fill_pipeline,
            unified_fill_pipeline,
            map_unified_fill_pipeline,
            layered_fill_pipeline,
            map_layered_fill_pipeline,
            map_edge_pipeline,
            colored_line_pipeline,
            surface_line_pipeline,
            uniform_buffer,
            layer_uniform_buffer,
            bind_group,
            layered_bind_group,
            depth_view,
        }
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        let (_, depth_view) = create_depth_texture(device, width, height);
        self.depth_view = depth_view;
    }

    /// Update the layer uniforms for noise/features visualization.
    pub fn set_layer_uniforms(&self, gpu: &GpuContext, layer_uniforms: &LayerUniforms) {
        gpu.queue.write_buffer(
            &self.layer_uniform_buffer,
            0,
            bytemuck::bytes_of(layer_uniforms),
        );
    }

    pub fn render(&mut self, gpu: &mut GpuContext, uniforms: &Uniforms, scene: RenderScene<'_>) {
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));

        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost) => {
                gpu.resize(gpu.size);
                self.resize(&gpu.device, gpu.size.width, gpu.size.height);
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

        let mut encoder = gpu
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
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let (fill_pipeline, bind_group) = match scene.fill_pipeline {
                FillPipelineKind::Globe => (&self.fill_pipeline, &self.bind_group),
                FillPipelineKind::Map => (&self.map_fill_pipeline, &self.bind_group),
                FillPipelineKind::UnifiedGlobe => (&self.unified_fill_pipeline, &self.bind_group),
                FillPipelineKind::UnifiedMap => (&self.map_unified_fill_pipeline, &self.bind_group),
                FillPipelineKind::LayeredGlobe => {
                    (&self.layered_fill_pipeline, &self.layered_bind_group)
                }
                FillPipelineKind::LayeredMap => {
                    (&self.map_layered_fill_pipeline, &self.layered_bind_group)
                }
            };

            // Track if we're using layered pipeline (needs bind group reset for other draws)
            let using_layered = matches!(
                scene.fill_pipeline,
                FillPipelineKind::LayeredGlobe | FillPipelineKind::LayeredMap
            );

            render_pass.set_pipeline(fill_pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.set_vertex_buffer(0, scene.fill.vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(scene.fill.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..scene.fill.index_count, 0, 0..1);

            // Reset to regular bind group if we used layered pipeline
            // (subsequent draws use pipelines expecting single-binding layout)
            if using_layered {
                render_pass.set_bind_group(0, &self.bind_group, &[]);
            }

            if let Some(edges) = scene.edges {
                match edges {
                    EdgeDraw::GlobeColored(draw) => {
                        render_pass.set_pipeline(&self.colored_line_pipeline);
                        render_pass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
                        render_pass.draw(0..draw.vertex_count, 0..1);
                    }
                    EdgeDraw::MapIndexed {
                        vertex_buffer,
                        index_buffer,
                        index_count,
                    } => {
                        render_pass.set_pipeline(&self.map_edge_pipeline);
                        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        render_pass
                            .set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..index_count, 0, 0..1);
                    }
                }
            }

            if let Some(arrows) = scene.arrows {
                render_pass.set_pipeline(&self.colored_line_pipeline);
                render_pass.set_vertex_buffer(0, arrows.vertex_buffer.slice(..));
                render_pass.draw(0..arrows.vertex_count, 0..1);
            }

            if let Some(pole_markers) = scene.pole_markers {
                render_pass.set_pipeline(&self.fill_pipeline);
                render_pass.set_vertex_buffer(0, pole_markers.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    pole_markers.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..pole_markers.index_count, 0, 0..1);
            }

            if let Some(rivers) = scene.rivers {
                render_pass.set_pipeline(&self.surface_line_pipeline);
                render_pass.set_vertex_buffer(0, rivers.vertex_buffer.slice(..));
                render_pass.draw(0..rivers.vertex_count, 0..1);
            }

            // Triangle-based rivers (uses unified pipeline for material-aware lighting)
            if let Some(river_mesh) = scene.river_mesh {
                render_pass.set_pipeline(&self.unified_fill_pipeline);
                render_pass.set_vertex_buffer(0, river_mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(river_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..river_mesh.index_count, 0, 0..1);
            }

            // Wind particle trails
            if let Some(wind_particles) = scene.wind_particles {
                render_pass.set_pipeline(&self.colored_line_pipeline);
                render_pass.set_vertex_buffer(0, wind_particles.vertex_buffer.slice(..));
                render_pass.draw(0..wind_particles.vertex_count, 0..1);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
