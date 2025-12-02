use wgpu::{util::DeviceExt, Buffer, BufferUsages, Device};

/// Create a vertex buffer from data.
pub fn create_vertex_buffer<T: bytemuck::Pod>(device: &Device, data: &[T], label: &str) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: BufferUsages::VERTEX,
    })
}

/// Create an index buffer from data.
pub fn create_index_buffer(device: &Device, data: &[u32], label: &str) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: BufferUsages::INDEX,
    })
}

/// Create a uniform buffer from data.
pub fn create_uniform_buffer<T: bytemuck::Pod>(device: &Device, data: &T, label: &str) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}
