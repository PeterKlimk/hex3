use glam::{Mat4, Vec3};

use crate::world::RELIEF_SCALE;

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
    // WGSL vec3 has 16-byte alignment, so _padding2 in shader starts at offset 112
    // We need: 64 + 16 + 16 + 4 + (12 align) + 12 + 4 = 128 bytes total
    // Simplest: use 7 f32s (28 bytes) to reach 128 from current offset 100
    pub _padding2: [f32; 7],
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
            _padding2: [0.0; 7],
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
