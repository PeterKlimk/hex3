use wgpu::{
    BindGroupLayout, BlendState, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
    DepthStencilState, Device, Face, FragmentState, FrontFace, MultisampleState,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPipeline,
    RenderPipelineDescriptor, ShaderModule, StencilState, TextureFormat, VertexBufferLayout,
    VertexState,
};

/// Builder for creating render pipelines with less boilerplate.
pub struct PipelineBuilder<'a> {
    device: &'a Device,
    shader: Option<&'a ShaderModule>,
    vertex_layouts: Vec<VertexBufferLayout<'a>>,
    bind_group_layouts: Vec<&'a BindGroupLayout>,
    format: TextureFormat,
    topology: PrimitiveTopology,
    polygon_mode: PolygonMode,
    cull_mode: Option<Face>,
    depth_format: Option<TextureFormat>,
    depth_write: bool,
    depth_bias: f32,
    label: Option<&'a str>,
}

impl<'a> PipelineBuilder<'a> {
    pub fn new(device: &'a Device, format: TextureFormat) -> Self {
        Self {
            device,
            shader: None,
            vertex_layouts: Vec::new(),
            bind_group_layouts: Vec::new(),
            format,
            topology: PrimitiveTopology::TriangleList,
            polygon_mode: PolygonMode::Fill,
            cull_mode: Some(Face::Back),
            depth_format: Some(TextureFormat::Depth32Float),
            depth_write: true,
            depth_bias: 0.0,
            label: None,
        }
    }

    pub fn shader(mut self, shader: &'a ShaderModule) -> Self {
        self.shader = Some(shader);
        self
    }

    pub fn vertex_layout(mut self, layout: VertexBufferLayout<'a>) -> Self {
        self.vertex_layouts.push(layout);
        self
    }

    pub fn bind_group_layout(mut self, layout: &'a BindGroupLayout) -> Self {
        self.bind_group_layouts.push(layout);
        self
    }

    pub fn topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn polygon_mode(mut self, mode: PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    pub fn cull_mode(mut self, mode: Option<Face>) -> Self {
        self.cull_mode = mode;
        self
    }

    pub fn depth_write(mut self, write: bool) -> Self {
        self.depth_write = write;
        self
    }

    pub fn depth_bias(mut self, bias: f32) -> Self {
        self.depth_bias = bias;
        self
    }

    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn build(self) -> RenderPipeline {
        let shader = self.shader.expect("Shader required");

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: self.label,
                bind_group_layouts: &self.bind_group_layouts,
                push_constant_ranges: &[],
            });

        self.device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: self.label,
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &self.vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(ColorTargetState {
                        format: self.format,
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: PrimitiveState {
                    topology: self.topology,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: self.cull_mode,
                    polygon_mode: self.polygon_mode,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: self.depth_format.map(|format| DepthStencilState {
                    format,
                    depth_write_enabled: self.depth_write,
                    depth_compare: CompareFunction::Less,
                    stencil: StencilState::default(),
                    bias: DepthBiasState {
                        constant: self.depth_bias as i32,
                        slope_scale: self.depth_bias,
                        clamp: 0.0,
                    },
                }),
                multisample: MultisampleState::default(),
                multiview: None,
                cache: None,
            })
    }
}

/// Create a depth texture.
pub fn create_depth_texture(
    device: &Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}
