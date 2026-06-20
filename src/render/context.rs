use std::sync::Arc;
use wgpu::{
    Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, PresentMode, Queue, RequestAdapterOptions, Surface, SurfaceConfiguration,
    TextureFormat, TextureUsages,
};
use winit::{dpi::PhysicalSize, window::Window};

/// GPU context holding all wgpu resources. `surface` is `None` for a headless
/// context (offscreen rendering, e.g. the sweep harness), which renders to its
/// own textures instead of a swapchain.
pub struct GpuContext {
    pub surface: Option<Surface<'static>>,
    pub device: Device,
    pub queue: Queue,
    pub config: SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    pub format: TextureFormat,
}

impl GpuContext {
    /// Create a new GPU context for the given window.
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // Create wgpu instance
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        // Create surface
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");

        // Request adapter
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable GPU adapter");

        // Raise the buffer-size cap from wgpu's conservative 256 MiB default to
        // what the adapter actually supports. The full-resolution fine mesh
        // produces a unified vertex buffer near/over 256 MiB (~5M shared Voronoi
        // vertices); desktop GPUs handle multi-GB buffers, so request the max.
        let mut required_limits = Limits::default();
        required_limits.max_buffer_size = adapter.limits().max_buffer_size;

        // Request device and queue
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("hex3_device"),
                required_features: Features::empty(),
                required_limits,
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            })
            .await
            .expect("Failed to create device");

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::AutoNoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self {
            surface: Some(surface),
            device,
            queue,
            config,
            size,
            format,
        }
    }

    /// Create a headless GPU context (no window/surface) for offscreen rendering.
    /// The format is a fixed sRGB 8-bit RGBA so readback bytes are PNG-ready.
    pub async fn new_headless(width: u32, height: u32) -> Self {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable GPU adapter");

        let mut required_limits = Limits::default();
        required_limits.max_buffer_size = adapter.limits().max_buffer_size;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("hex3_headless_device"),
                required_features: Features::empty(),
                required_limits,
                memory_hints: Default::default(),
                trace: Default::default(),
                experimental_features: Default::default(),
            })
            .await
            .expect("Failed to create device");

        let format = TextureFormat::Rgba8UnormSrgb;
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        Self {
            surface: None,
            device,
            queue,
            config,
            size: PhysicalSize::new(width, height),
            format,
        }
    }

    /// Resize the surface (no-op for a headless context).
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            if let Some(surface) = &self.surface {
                surface.configure(&self.device, &self.config);
            }
        }
    }

    /// Get aspect ratio.
    pub fn aspect(&self) -> f32 {
        self.size.width as f32 / self.size.height as f32
    }
}
