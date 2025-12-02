use glam::{Mat4, Vec3};

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
    pub _padding2: f32,
}

impl Uniforms {
    pub fn new(view_proj: Mat4, camera_pos: Vec3, light_dir: Vec3) -> Self {
        Self {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding1: 0.0,
            light_dir: light_dir.normalize().to_array(),
            _padding2: 0.0,
        }
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
