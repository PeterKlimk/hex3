use glam::{Mat4, Vec3};
use std::f32::consts::FRAC_PI_4;

/// Maximum tilt angle (like a globe on a stand - can't flip over).
const MAX_TILT: f32 = 1.4; // ~80 degrees up or down

/// An orbit camera that rotates around a target point.
/// Behaves like a globe: free horizontal spin, limited vertical tilt.
pub struct OrbitCamera {
    /// Point the camera looks at.
    pub target: Vec3,
    /// Distance from the target.
    pub distance: f32,
    /// Horizontal rotation angle (radians) - unlimited.
    pub yaw: f32,
    /// Vertical tilt angle (radians) - limited like a globe.
    pub pitch: f32,
    /// Vertical field of view (radians).
    pub fov_y: f32,
    /// Aspect ratio (width / height).
    pub aspect: f32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 3.0,
            yaw: 0.0,
            pitch: 0.4,       // Slight tilt to see the sphere nicely
            fov_y: FRAC_PI_4, // 45 degrees
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl OrbitCamera {
    /// Create a new orbit camera.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the camera's eye position in world space.
    pub fn eye_position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + Vec3::new(x, y, z)
    }

    /// Get the view matrix.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye_position(), self.target, Vec3::Y)
    }

    /// Get the projection matrix.
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Get the combined view-projection matrix.
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    /// Rotate the camera by the given delta angles.
    /// Yaw (horizontal) is unlimited, pitch (vertical) is clamped like a globe.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        // Clamp pitch to globe-like tilt range
        self.pitch = (self.pitch + delta_pitch).clamp(-MAX_TILT, MAX_TILT);
    }

    /// Zoom the camera by changing distance.
    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta).clamp(1.5, 10.0);
    }

    /// Update the aspect ratio.
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }
}

/// Camera controller for handling input.
pub struct CameraController {
    /// Rotation sensitivity (radians per pixel).
    pub rotate_sensitivity: f32,
    /// Zoom sensitivity (units per scroll).
    pub zoom_sensitivity: f32,
    /// Whether the mouse is currently pressed for rotation.
    pub is_rotating: bool,
    /// Last mouse position.
    pub last_mouse: Option<(f32, f32)>,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            rotate_sensitivity: 0.005,
            zoom_sensitivity: 0.3,
            is_rotating: false,
            last_mouse: None,
        }
    }
}

impl CameraController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Handle mouse button press.
    pub fn on_mouse_press(&mut self) {
        self.is_rotating = true;
        // Don't set last_mouse here - let first move event set it
        // This prevents jumping when clicking
    }

    /// Handle mouse button release.
    pub fn on_mouse_release(&mut self) {
        self.is_rotating = false;
        self.last_mouse = None;
    }

    /// Handle mouse movement. Returns true if the camera should be updated.
    pub fn on_mouse_move(&mut self, x: f32, y: f32, camera: &mut OrbitCamera) -> bool {
        if !self.is_rotating {
            return false;
        }

        let moved = if let Some((last_x, last_y)) = self.last_mouse {
            let delta_x = x - last_x;
            let delta_y = y - last_y;
            camera.orbit(
                -delta_x * self.rotate_sensitivity,
                delta_y * self.rotate_sensitivity, // Flip vertical
            );
            true
        } else {
            // First move after press - just record position, don't rotate
            false
        };
        self.last_mouse = Some((x, y));
        moved
    }

    /// Handle scroll. Returns true if the camera should be updated.
    pub fn on_scroll(&mut self, delta: f32, camera: &mut OrbitCamera) -> bool {
        camera.zoom(delta * self.zoom_sensitivity);
        true
    }
}
