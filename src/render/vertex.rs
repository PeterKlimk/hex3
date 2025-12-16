use wgpu::{BufferAddress, VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

use crate::geometry::{LayeredVertex, MeshVertex, SurfaceVertex, UnifiedVertex};

impl UnifiedVertex {
    /// Describe the vertex buffer layout for wgpu.
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<UnifiedVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                // Position (vec3)
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                // Normal (vec3)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                },
                // Color (vec3)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as BufferAddress,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
                // Elevation (f32)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 9]>() as BufferAddress,
                    shader_location: 3,
                    format: VertexFormat::Float32,
                },
                // Material (u32)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 10]>() as BufferAddress,
                    shader_location: 4,
                    format: VertexFormat::Uint32,
                },
                // Wrap offset (f32) - for map view antimeridian handling
                VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 10]>() + std::mem::size_of::<u32>())
                        as BufferAddress,
                    shader_location: 5,
                    format: VertexFormat::Float32,
                },
            ],
        }
    }
}

impl MeshVertex {
    /// Describe the vertex buffer layout for wgpu.
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                // Position (vec3)
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                // Normal (vec3)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Float32x3,
                },
                // Color (vec3)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as BufferAddress,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
                // Wrap offset (f32) - for map view antimeridian handling
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 9]>() as BufferAddress,
                    shader_location: 3,
                    format: VertexFormat::Float32,
                },
            ],
        }
    }
}

impl SurfaceVertex {
    /// Describe the vertex buffer layout for wgpu.
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<SurfaceVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                // Position (vec3)
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                // Elevation (f32)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Float32,
                },
                // Color with alpha (vec4)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as BufferAddress,
                    shader_location: 2,
                    format: VertexFormat::Float32x4,
                },
            ],
        }
    }
}

impl LayeredVertex {
    /// Describe the vertex buffer layout for wgpu.
    ///
    /// Layout: position (vec3), wrap_offset (f32), normal (vec3), _padding (f32), layers[0..6]
    /// Shader locations: 0=position, 1=wrap_offset, 2=normal, 3=layers[0..4], 4=layer4, 5=layer5
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<LayeredVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                // Position (vec3) at offset 0
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x3,
                },
                // Wrap offset (f32) at offset 12 - for map view antimeridian handling
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as BufferAddress,
                    shader_location: 1,
                    format: VertexFormat::Float32,
                },
                // Normal (vec3) at offset 16
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as BufferAddress,
                    shader_location: 2,
                    format: VertexFormat::Float32x3,
                },
                // Layers 0-3 (vec4) at offset 32 (skipping _padding at 28)
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as BufferAddress,
                    shader_location: 3,
                    format: VertexFormat::Float32x4,
                },
                // Layer 4 (f32) at offset 48
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 12]>() as BufferAddress,
                    shader_location: 4,
                    format: VertexFormat::Float32,
                },
                // Layer 5 (f32) at offset 52
                VertexAttribute {
                    offset: std::mem::size_of::<[f32; 13]>() as BufferAddress,
                    shader_location: 5,
                    format: VertexFormat::Float32,
                },
            ],
        }
    }
}
