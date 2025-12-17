//! Cubemap texture helpers.
//!
//! WebGPU cubemaps are represented as 2D array textures with 6 layers.
//! This module centralizes creation of the texture and its common views:
//! - a `D2Array` view (for explicit face sampling),
//! - a `Cube` view (optional, for `texture_cube` sampling),
//! - and per-face `D2` views for rendering.

/// Number of faces in a cubemap.
pub const CUBEMAP_FACE_COUNT: u32 = 6;

/// A cubemap texture with commonly used views.
pub struct CubemapTexture {
    pub texture: wgpu::Texture,
    pub array_view: wgpu::TextureView,
    pub cube_view: wgpu::TextureView,
    pub face_views: [wgpu::TextureView; 6],
    pub sampler: wgpu::Sampler,
}

impl CubemapTexture {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        face_size: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        sampler: wgpu::SamplerDescriptor<'_>,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: CUBEMAP_FACE_COUNT,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });

        let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(CUBEMAP_FACE_COUNT),
            ..Default::default()
        });

        let face_views: [wgpu::TextureView; 6] = std::array::from_fn(|i| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(label),
            ..sampler
        });

        Self {
            texture,
            array_view,
            cube_view,
            face_views,
            sampler,
        }
    }
}
