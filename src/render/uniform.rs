use glam::{Mat4, Vec3};

use crate::world::RELIEF_SCALE;

/// Uniforms for layered visualization (noise/features modes).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerUniforms {
    /// Which layer to display (0-5).
    pub layer_index: u32,
    /// Colormap mode: 0 = noise, 1 = features.
    pub colormap_mode: u32,
    pub _padding: [u32; 2],
}

impl LayerUniforms {
    pub fn noise(layer_index: u32) -> Self {
        Self {
            layer_index,
            colormap_mode: 0,
            _padding: [0; 2],
        }
    }

    pub fn features(layer_index: u32) -> Self {
        Self {
            layer_index,
            colormap_mode: 1,
            _padding: [0; 2],
        }
    }
}

/// Uniforms for the main shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    /// View-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space.
    pub camera_pos: [f32; 3],
    pub _padding1: f32,
    /// Light direction (normalized).
    pub light_dir: [f32; 3],
    /// Relief scale (0.0 = flat, RELIEF_SCALE = 3D terrain).
    pub relief_scale: f32,
    /// Hemisphere lighting toggle (1.0 = enabled, 0.0 = simple diffuse).
    pub hemisphere_lighting: f32,
    /// Map mode (0.0 = globe view, 1.0 = equirectangular map view).
    pub map_mode: f32,
    /// Padding to align to 16 bytes.
    pub _padding2: [f32; 6],
}

impl Uniforms {
    pub fn new(view_proj: Mat4, camera_pos: Vec3, light_dir: Vec3) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding1: 0.0,
            light_dir: light_dir.normalize().to_array(),
            relief_scale: 0.0,
            hemisphere_lighting: 1.0,
            map_mode: 0.0,
            _padding2: [0.0; 6],
        }
    }

    /// Set whether relief displacement is enabled.
    pub fn with_relief(mut self, enabled: bool) -> Self {
        self.relief_scale = if enabled { RELIEF_SCALE } else { 0.0 };
        self
    }

    /// Set whether hemisphere lighting is enabled.
    pub fn with_hemisphere_lighting(mut self, enabled: bool) -> Self {
        self.hemisphere_lighting = if enabled { 1.0 } else { 0.0 };
        self
    }

    /// Set whether map mode is enabled.
    pub fn with_map_mode(mut self, enabled: bool) -> Self {
        self.map_mode = if enabled { 1.0 } else { 0.0 };
        self
    }
}

impl Default for Uniforms {
    fn default() -> Self {
        Self::new(
            Mat4::IDENTITY,
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.5, 1.0, 0.3).normalize(),
        )
    }
}
