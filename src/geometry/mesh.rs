use glam::Vec3;
use std::collections::HashSet;

use super::SphericalVoronoi;

/// Material types for unified rendering.
/// Each material responds differently to lighting.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Material {
    /// Matte terrain - diffuse lighting only
    Land = 0,
    /// Deep water - diffuse + specular + fresnel
    Ocean = 1,
    /// Shallow water - diffuse + specular + fresnel (subtler)
    Lake = 2,
    /// River water - diffuse + slight transparency
    River = 3,
    /// Ice/snow - diffuse + sharp specular (future)
    IceSnow = 4,
}

/// A unified vertex for material-aware rendering.
///
/// All visible geometry (terrain, water, rivers) uses this format,
/// with the material attribute controlling lighting behavior in the shader.
///
/// Position is stored on the unit sphere (undisplaced). The shader applies
/// displacement based on elevation and procedural micro noise.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UnifiedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    /// Simulation elevation (without micro noise). Shader adds micro noise.
    pub elevation: f32,
    pub material: u32,
    /// Wrap offset for antimeridian handling in map view.
    /// Added to x coordinate when projecting to equirectangular.
    pub wrap_offset: f32,
}

impl UnifiedVertex {
    pub fn new(
        position: Vec3,
        normal: Vec3,
        color: Vec3,
        elevation: f32,
        material: Material,
    ) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            color: color.to_array(),
            elevation,
            material: material as u32,
            wrap_offset: 0.0,
        }
    }

    /// Create with explicit wrap offset for antimeridian handling.
    pub fn with_wrap_offset(mut self, wrap_offset: f32) -> Self {
        self.wrap_offset = wrap_offset;
        self
    }

    /// Create a land vertex (most common case)
    pub fn land(position: Vec3, normal: Vec3, color: Vec3, elevation: f32) -> Self {
        Self::new(position, normal, color, elevation, Material::Land)
    }

    /// Create an ocean vertex
    pub fn ocean(position: Vec3, normal: Vec3, color: Vec3, elevation: f32) -> Self {
        Self::new(position, normal, color, elevation, Material::Ocean)
    }

    /// Create a lake vertex
    pub fn lake(position: Vec3, normal: Vec3, color: Vec3, elevation: f32) -> Self {
        Self::new(position, normal, color, elevation, Material::Lake)
    }

    /// Create a river vertex
    pub fn river(position: Vec3, normal: Vec3, color: Vec3, elevation: f32) -> Self {
        Self::new(position, normal, color, elevation, Material::River)
    }
}

/// A mesh vertex with position, normal, and color.
/// TODO: Deprecate in favor of UnifiedVertex
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    /// Wrap offset for antimeridian handling in map view.
    pub wrap_offset: f32,
}

impl MeshVertex {
    pub fn new(position: Vec3, normal: Vec3, color: Vec3) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            color: color.to_array(),
            wrap_offset: 0.0,
        }
    }

    /// Create with explicit wrap offset for antimeridian handling.
    pub fn with_wrap_offset(mut self, wrap_offset: f32) -> Self {
        self.wrap_offset = wrap_offset;
        self
    }
}

/// A vertex for surface features (rivers, roads, etc.) that can be displaced by relief.
///
/// The shader will displace this vertex by: `normalize(position) * (1.0 + elevation * relief_scale)`
/// where `relief_scale` is a uniform (0.0 = flat, RELIEF_SCALE = 3D relief).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceVertex {
    /// Base position on unit sphere
    pub position: [f32; 3],
    /// Terrain elevation at this point (for displacement)
    pub elevation: f32,
    /// Vertex color with alpha (RGBA)
    pub color: [f32; 4],
}

impl SurfaceVertex {
    pub fn new(position: Vec3, elevation: f32, color: Vec3, alpha: f32) -> Self {
        Self {
            position: position.to_array(),
            elevation,
            color: [color.x, color.y, color.z, alpha],
        }
    }
}

/// A vertex for layered visualization (noise layers, feature fields).
///
/// Stores 6 scalar values that can be selected via shader uniform,
/// enabling instant layer switching without buffer regeneration.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayeredVertex {
    /// Position on unit sphere
    pub position: [f32; 3],
    /// Wrap offset for antimeridian handling in map view.
    pub wrap_offset: f32,
    /// Surface normal
    pub normal: [f32; 3],
    /// Padding for alignment
    pub _padding: f32,
    /// 6 layer values (selected by uniform in shader)
    pub layers: [f32; 6],
}

impl LayeredVertex {
    pub fn new(position: Vec3, normal: Vec3, layers: [f32; 6]) -> Self {
        Self {
            position: position.to_array(),
            wrap_offset: 0.0,
            normal: normal.to_array(),
            _padding: 0.0,
            layers,
        }
    }

    /// Create with explicit wrap offset for antimeridian handling.
    pub fn with_wrap_offset(mut self, wrap_offset: f32) -> Self {
        self.wrap_offset = wrap_offset;
        self
    }
}

/// A mesh with layered data for visualization modes.
pub struct LayeredMesh {
    /// Vertices with layer data.
    pub vertices: Vec<LayeredVertex>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
}

impl LayeredMesh {
    /// Generate a layered mesh from a Voronoi diagram.
    ///
    /// `layer_fn` returns 6 layer values for each cell.
    pub fn from_voronoi<F>(voronoi: &SphericalVoronoi, layer_fn: F) -> Self
    where
        F: Fn(usize) -> [f32; 6],
    {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for cell in voronoi.iter_cells() {
            if cell.len() < 3 {
                continue;
            }

            let generator = voronoi.generators[cell.generator_index];
            let normal = generator; // On unit sphere, normal = position
            let layers = layer_fn(cell.generator_index);

            // Compute wrap offsets for this cell (for map view antimeridian handling)
            // Include generator (center) and all perimeter vertices
            let mut cell_positions: Vec<Vec3> = vec![generator];
            cell_positions.extend(
                cell.vertex_indices
                    .iter()
                    .map(|&vi| voronoi.vertices[vi as usize]),
            );
            let wrap_offsets = compute_cell_wrap_offsets(&cell_positions);

            // Fan triangulation from the generator point
            let base_idx = vertices.len() as u32;

            // Center vertex (at generator)
            let center_wrap = wrap_offsets.as_ref().map(|w| w[0]).unwrap_or(0.0);
            vertices
                .push(LayeredVertex::new(generator, normal, layers).with_wrap_offset(center_wrap));

            // Perimeter vertices
            for (local_idx, &vi) in cell.vertex_indices.iter().enumerate() {
                let v = voronoi.vertices[vi as usize];
                let wrap_offset = wrap_offsets
                    .as_ref()
                    .map(|w| w[local_idx + 1]) // +1 because generator is at index 0
                    .unwrap_or(0.0);
                vertices.push(LayeredVertex::new(v, normal, layers).with_wrap_offset(wrap_offset));
            }

            // Triangles (fan from center)
            let n = cell.vertex_indices.len() as u32;
            for i in 0..n {
                indices.push(base_idx); // center
                indices.push(base_idx + 1 + i);
                indices.push(base_idx + 1 + (i + 1) % n);
            }
        }

        Self { vertices, indices }
    }

    /// Project vertices to 2D equirectangular map coordinates.
    pub fn to_map_vertices(&self) -> (Vec<LayeredVertex>, Vec<u32>) {
        // First pass: project all vertices
        let projected: Vec<(f32, f32)> = self
            .vertices
            .iter()
            .map(|v| {
                let p = Vec3::from_array(v.position);
                sphere_to_equirectangular(p)
            })
            .collect();

        // Second pass: identify cell boundaries from triangle structure
        let mut cell_ranges: Vec<(usize, usize)> = Vec::new();
        let mut current_fan_center: Option<u32> = None;
        let mut cell_start = 0;

        for (tri_idx, tri) in self.indices.chunks(3).enumerate() {
            let fan_center = tri[0];

            match current_fan_center {
                None => {
                    current_fan_center = Some(fan_center);
                }
                Some(prev_center) => {
                    if fan_center != prev_center {
                        cell_ranges.push((cell_start, tri_idx));
                        cell_start = tri_idx;
                        current_fan_center = Some(fan_center);
                    }
                }
            }
        }

        // Don't forget the last cell
        if !self.indices.is_empty() {
            cell_ranges.push((cell_start, self.indices.len() / 3));
        }

        // Third pass: build output with per-cell wrapping correction
        let mut new_vertices = Vec::with_capacity(self.vertices.len());
        let mut new_indices = Vec::with_capacity(self.indices.len());

        for (cell_start_tri, cell_end_tri) in cell_ranges {
            // Collect all vertex indices used by this cell
            let mut cell_vertex_indices: Vec<u32> = Vec::new();
            for tri_idx in cell_start_tri..cell_end_tri {
                let i = tri_idx * 3;
                cell_vertex_indices.push(self.indices[i]);
                cell_vertex_indices.push(self.indices[i + 1]);
                cell_vertex_indices.push(self.indices[i + 2]);
            }
            cell_vertex_indices.sort_unstable();
            cell_vertex_indices.dedup();

            // Get projected x coords for this cell
            let x_coords: Vec<f32> = cell_vertex_indices
                .iter()
                .map(|&idx| projected[idx as usize].0)
                .collect();

            let min_x = x_coords.iter().copied().fold(f32::INFINITY, f32::min);
            let max_x = x_coords.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let x_span = max_x - min_x;

            // If span > 1.0, the cell crosses the antimeridian
            let needs_wrap = x_span > 1.0;

            // Map old vertex index -> new vertex index for this cell
            let base_new_idx = new_vertices.len() as u32;
            let mut old_to_new: std::collections::HashMap<u32, u32> =
                std::collections::HashMap::new();

            for (local_idx, &old_idx) in cell_vertex_indices.iter().enumerate() {
                old_to_new.insert(old_idx, base_new_idx + local_idx as u32);

                let (mut x, y) = projected[old_idx as usize];

                if needs_wrap && x < 0.0 {
                    x += 2.0;
                }

                let old_v = &self.vertices[old_idx as usize];
                new_vertices.push(LayeredVertex {
                    position: [x, y, 0.0],
                    wrap_offset: 0.0,
                    normal: [0.0, 0.0, 1.0],
                    _padding: 0.0,
                    layers: old_v.layers,
                });
            }

            // Emit triangles with remapped indices
            for tri_idx in cell_start_tri..cell_end_tri {
                let i = tri_idx * 3;
                new_indices.push(old_to_new[&self.indices[i]]);
                new_indices.push(old_to_new[&self.indices[i + 1]]);
                new_indices.push(old_to_new[&self.indices[i + 2]]);
            }
        }

        (new_vertices, new_indices)
    }

    /// Project vertices using precomputed projection data.
    ///
    /// This is faster than `to_map_vertices` when projecting multiple
    /// meshes with the same topology.
    pub fn to_map_vertices_with_projection(
        &self,
        projection: &MapProjection,
    ) -> (Vec<LayeredVertex>, Vec<u32>) {
        let mut new_vertices = Vec::with_capacity(self.vertices.len());
        let mut new_indices = Vec::with_capacity(self.indices.len());

        for &(cell_start_tri, cell_end_tri) in &projection.cell_ranges {
            // Collect all vertex indices used by this cell
            let mut cell_vertex_indices: Vec<u32> = Vec::new();
            for tri_idx in cell_start_tri..cell_end_tri {
                let i = tri_idx * 3;
                cell_vertex_indices.push(self.indices[i]);
                cell_vertex_indices.push(self.indices[i + 1]);
                cell_vertex_indices.push(self.indices[i + 2]);
            }
            cell_vertex_indices.sort_unstable();
            cell_vertex_indices.dedup();

            // Get projected x coords for this cell
            let x_coords: Vec<f32> = cell_vertex_indices
                .iter()
                .map(|&idx| projection.projected[idx as usize].0)
                .collect();

            let min_x = x_coords.iter().copied().fold(f32::INFINITY, f32::min);
            let max_x = x_coords.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let x_span = max_x - min_x;

            // If span > 1.0, the cell crosses the antimeridian
            let needs_wrap = x_span > 1.0;

            // Map old vertex index -> new vertex index for this cell
            let base_new_idx = new_vertices.len() as u32;
            let mut old_to_new: std::collections::HashMap<u32, u32> =
                std::collections::HashMap::new();

            for (local_idx, &old_idx) in cell_vertex_indices.iter().enumerate() {
                old_to_new.insert(old_idx, base_new_idx + local_idx as u32);

                let (mut x, y) = projection.projected[old_idx as usize];

                if needs_wrap && x < 0.0 {
                    x += 2.0;
                }

                let old_v = &self.vertices[old_idx as usize];
                new_vertices.push(LayeredVertex {
                    position: [x, y, 0.0],
                    wrap_offset: 0.0,
                    normal: [0.0, 0.0, 1.0],
                    _padding: 0.0,
                    layers: old_v.layers,
                });
            }

            // Emit triangles with remapped indices
            for tri_idx in cell_start_tri..cell_end_tri {
                let i = tri_idx * 3;
                new_indices.push(old_to_new[&self.indices[i]]);
                new_indices.push(old_to_new[&self.indices[i + 1]]);
                new_indices.push(old_to_new[&self.indices[i + 2]]);
            }
        }

        (new_vertices, new_indices)
    }

    /// Compute map projection data from this mesh.
    pub fn compute_projection(&self) -> MapProjection {
        let positions: Vec<[f32; 3]> = self.vertices.iter().map(|v| v.position).collect();
        MapProjection::compute(&positions, &self.indices)
    }
}

/// A renderable mesh generated from a spherical Voronoi diagram.
pub struct VoronoiMesh {
    /// Vertices for the filled triangles.
    pub vertices: Vec<MeshVertex>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
    /// Edge indices for wireframe (groups of 2).
    pub edge_indices: Vec<u32>,
}

/// A unified mesh with material-aware vertices.
/// Uses UnifiedVertex for proper lighting based on material type.
pub struct UnifiedMesh {
    /// Vertices with material information.
    pub vertices: Vec<UnifiedVertex>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
    /// Edge indices for wireframe (groups of 2).
    pub edge_indices: Vec<u32>,
}

impl UnifiedMesh {
    /// Generate a unified mesh with color and material per cell.
    pub fn from_voronoi<C, M>(voronoi: &SphericalVoronoi, color_fn: C, material_fn: M) -> Self
    where
        C: Fn(usize) -> Vec3,
        M: Fn(usize) -> Material,
    {
        Self::from_voronoi_with_elevation(voronoi, color_fn, material_fn, |_| 0.0)
    }

    /// Generate a unified mesh with elevation, color, and material per cell.
    ///
    /// Positions are stored on the unit sphere (undisplaced). The shader applies
    /// displacement based on the elevation attribute and procedural micro noise.
    ///
    /// Elevation handling at boundaries:
    /// - Water cells (Ocean, Lake): vertices are flat at water_level (no averaging)
    /// - Land cells: vertices are averaged, but clamped to water_level at water boundaries
    /// This ensures water is perfectly flat while land slopes down to meet it.
    pub fn from_voronoi_with_elevation<C, M, E>(
        voronoi: &SphericalVoronoi,
        color_fn: C,
        material_fn: M,
        elevation_fn: E,
    ) -> Self
    where
        C: Fn(usize) -> Vec3,
        M: Fn(usize) -> Material,
        E: Fn(usize) -> f32,
    {
        // Step 1: Precompute materials and elevations for all cells
        let cell_materials: Vec<Material> =
            (0..voronoi.num_cells()).map(|i| material_fn(i)).collect();
        let cell_elevations: Vec<f32> = (0..voronoi.num_cells()).map(|i| elevation_fn(i)).collect();

        // Step 2: For each vertex, compute elevation based on adjacent cell types
        // Track: sum of land elevations, count of land cells, water_level if any water adjacent
        let mut vertex_land_sum = vec![0.0f32; voronoi.vertices.len()];
        let mut vertex_land_count = vec![0u32; voronoi.vertices.len()];
        let mut vertex_water_level = vec![None::<f32>; voronoi.vertices.len()];

        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            let material = cell_materials[cell_idx];
            let elevation = cell_elevations[cell_idx];
            let is_water = matches!(material, Material::Ocean | Material::Lake);

            for &vertex_idx in cell.vertex_indices {
                let vi = vertex_idx as usize;
                if is_water {
                    // Track water level (use max in case of adjacent lakes at different levels)
                    vertex_water_level[vi] = Some(
                        vertex_water_level[vi]
                            .map(|wl| wl.max(elevation))
                            .unwrap_or(elevation),
                    );
                } else {
                    // Accumulate land elevation for averaging
                    vertex_land_sum[vi] += elevation;
                    vertex_land_count[vi] += 1;
                }
            }
        }

        // Step 3: Compute final vertex elevations
        let vertex_elevations: Vec<f32> = (0..voronoi.vertices.len())
            .map(|v| {
                let water_level = vertex_water_level[v];
                let land_count = vertex_land_count[v];

                match (water_level, land_count) {
                    // All water: use water level
                    (Some(wl), 0) => wl,
                    // All land: average land elevations
                    (None, n) if n > 0 => vertex_land_sum[v] / n as f32,
                    // Mixed: use max of land average and water level (land meets water)
                    (Some(wl), n) if n > 0 => {
                        let land_avg = vertex_land_sum[v] / n as f32;
                        land_avg.max(wl)
                    }
                    // Shouldn't happen, but default to 0
                    _ => 0.0,
                }
            })
            .collect();

        // Step 4: Build mesh with undisplaced positions and elevation attribute
        // Shader will apply displacement: pos * (1 + elevation * relief_scale)
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut edge_set: HashSet<(u32, u32)> = HashSet::new();

        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            if cell.len() < 3 {
                continue;
            }

            let color = color_fn(cell_idx);
            let material = material_fn(cell_idx);

            // Compute wrap offsets for this cell (for map view antimeridian handling)
            let cell_positions: Vec<Vec3> = cell
                .vertex_indices
                .iter()
                .map(|&vi| voronoi.vertices[vi as usize])
                .collect();
            let wrap_offsets = compute_cell_wrap_offsets(&cell_positions);

            // Base index for this cell's vertices
            let base_idx = vertices.len() as u32;

            // Add vertices with undisplaced positions - shader does displacement
            // Water cells use flat water_level
            // Land cells use averaged elevation; at water boundaries, max(land_avg, water_level)
            let is_water = matches!(material, Material::Ocean | Material::Lake);
            let cell_water_level = cell_elevations[cell_idx];

            for (local_idx, &vertex_idx) in cell.vertex_indices.iter().enumerate() {
                let vi = vertex_idx as usize;
                let pos = voronoi.vertices[vi];
                let elevation = if is_water {
                    // Water cells are flat at their water level
                    cell_water_level
                } else if let Some(wl) = vertex_water_level[vi] {
                    // Land vertex touching water: use water level to meet seamlessly
                    wl
                } else {
                    // Interior land vertex: use averaged elevation
                    vertex_elevations[vi]
                };
                // Use sphere normal for smooth lighting (no terrain shadows)
                let wrap_offset = wrap_offsets.as_ref().map(|w| w[local_idx]).unwrap_or(0.0);
                vertices.push(
                    UnifiedVertex::new(pos, pos, color, elevation, material)
                        .with_wrap_offset(wrap_offset),
                );
            }

            // Fan triangulation from first vertex
            let n = cell.vertex_indices.len();
            for i in 1..n - 1 {
                indices.push(base_idx);
                indices.push(base_idx + i as u32);
                indices.push(base_idx + (i + 1) as u32);
            }

            // Collect edges
            for i in 0..n {
                let a = cell.vertex_indices[i] as u32;
                let b = cell.vertex_indices[(i + 1) % n] as u32;
                let edge = if a < b { (a, b) } else { (b, a) };
                edge_set.insert(edge);
            }
        }

        let edge_indices: Vec<u32> = edge_set.iter().flat_map(|&(a, b)| [a, b]).collect();

        UnifiedMesh {
            vertices,
            indices,
            edge_indices,
        }
    }

    /// Project vertices to 2D equirectangular map coordinates.
    pub fn to_map_vertices(&self) -> (Vec<UnifiedVertex>, Vec<u32>) {
        // First pass: project all vertices
        let projected: Vec<(f32, f32)> = self
            .vertices
            .iter()
            .map(|v| {
                let p = Vec3::from_array(v.position);
                sphere_to_equirectangular(p)
            })
            .collect();

        // Second pass: identify cell boundaries from triangle structure
        let mut cell_ranges: Vec<(usize, usize)> = Vec::new();
        let mut current_fan_center: Option<u32> = None;
        let mut cell_start = 0;

        for (tri_idx, tri) in self.indices.chunks(3).enumerate() {
            let fan_center = tri[0];

            match current_fan_center {
                None => {
                    current_fan_center = Some(fan_center);
                }
                Some(prev_center) => {
                    if fan_center != prev_center {
                        cell_ranges.push((cell_start, tri_idx));
                        cell_start = tri_idx;
                        current_fan_center = Some(fan_center);
                    }
                }
            }
        }
        let num_tris = self.indices.len() / 3;
        if cell_start < num_tris {
            cell_ranges.push((cell_start, num_tris));
        }

        // Third pass: compute wrap offset per cell
        let mut wrap_offset: Vec<f32> = vec![0.0; self.vertices.len()];

        for &(start_tri, end_tri) in &cell_ranges {
            let mut cell_vertex_indices: Vec<usize> = Vec::new();
            for tri_idx in start_tri..end_tri {
                let base = tri_idx * 3;
                cell_vertex_indices.push(self.indices[base] as usize);
                cell_vertex_indices.push(self.indices[base + 1] as usize);
                cell_vertex_indices.push(self.indices[base + 2] as usize);
            }
            cell_vertex_indices.sort();
            cell_vertex_indices.dedup();

            let xs: Vec<f32> = cell_vertex_indices
                .iter()
                .map(|&i| projected[i].0)
                .collect();
            let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            if max_x - min_x > 1.0 {
                let left_count = xs.iter().filter(|&&x| x < 0.0).count();
                let right_count = xs.len() - left_count;

                for &i in &cell_vertex_indices {
                    let x = projected[i].0;
                    if left_count > right_count {
                        if x > 0.5 {
                            wrap_offset[i] = -2.0;
                        }
                    } else if x < -0.5 {
                        wrap_offset[i] = 2.0;
                    }
                }
            }
        }

        // Fourth pass: build output with wrapped coordinates
        let mut out_vertices = Vec::new();
        let mut out_indices = Vec::new();

        for tri in self.indices.chunks(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let base = out_vertices.len() as u32;

            for &i in &[i0, i1, i2] {
                let (x, y) = projected[i];
                out_vertices.push(UnifiedVertex {
                    position: [x + wrap_offset[i], y, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    color: self.vertices[i].color,
                    elevation: 0.0, // Map view is flat
                    material: self.vertices[i].material,
                    wrap_offset: 0.0, // Not used in 2D map
                });
            }

            out_indices.push(base);
            out_indices.push(base + 1);
            out_indices.push(base + 2);
        }

        (out_vertices, out_indices)
    }

    /// Project vertices using precomputed projection data.
    ///
    /// This is much faster than `to_map_vertices` when projecting multiple
    /// meshes with the same topology, as the expensive cell boundary detection
    /// and wrap offset computation is done once and reused.
    pub fn to_map_vertices_with_projection(
        &self,
        projection: &MapProjection,
    ) -> (Vec<UnifiedVertex>, Vec<u32>) {
        let mut out_vertices = Vec::with_capacity(self.indices.len());
        let mut out_indices = Vec::with_capacity(self.indices.len());

        for tri in self.indices.chunks(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let base = out_vertices.len() as u32;

            for &i in &[i0, i1, i2] {
                let (x, y) = projection.projected[i];
                out_vertices.push(UnifiedVertex {
                    position: [x + projection.wrap_offset[i], y, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    color: self.vertices[i].color,
                    elevation: 0.0, // Map view is flat
                    material: self.vertices[i].material,
                    wrap_offset: 0.0, // Not used in 2D map
                });
            }

            out_indices.push(base);
            out_indices.push(base + 1);
            out_indices.push(base + 2);
        }

        (out_vertices, out_indices)
    }

    /// Compute map projection data from this mesh.
    pub fn compute_projection(&self) -> MapProjection {
        let positions: Vec<[f32; 3]> = self.vertices.iter().map(|v| v.position).collect();
        MapProjection::compute(&positions, &self.indices)
    }
}

/// Project a 3D sphere point to 2D equirectangular coordinates.
/// Returns (x, y) in range [-1, 1] for both axes.
pub fn sphere_to_equirectangular(p: Vec3) -> (f32, f32) {
    let lon = p.z.atan2(p.x); // -PI to PI
    let lat = p.y.asin(); // -PI/2 to PI/2

    let x = lon / std::f32::consts::PI; // -1 to 1
    let y = lat / std::f32::consts::FRAC_PI_2; // -1 to 1
    (x, y)
}

/// Compute wrap offsets for vertices in a cell that crosses the antimeridian.
///
/// Returns a Vec of wrap offsets (one per vertex in the cell), or None if the
/// cell doesn't need wrapping. Vertices near the "minority" side get offset
/// to join the majority.
pub fn compute_cell_wrap_offsets(positions: &[Vec3]) -> Option<Vec<f32>> {
    // Project all vertices to get x coordinates
    let x_coords: Vec<f32> = positions
        .iter()
        .map(|&p| sphere_to_equirectangular(p).0)
        .collect();

    let min_x = x_coords.iter().copied().fold(f32::INFINITY, f32::min);
    let max_x = x_coords.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Cell crosses antimeridian if it spans more than half the map
    if max_x - min_x <= 1.0 {
        return None; // No wrapping needed
    }

    // Count vertices on each side
    let left_count = x_coords.iter().filter(|&&x| x < 0.0).count();
    let right_count = x_coords.len() - left_count;

    // Compute wrap offset for each vertex
    let offsets: Vec<f32> = x_coords
        .iter()
        .map(|&x| {
            if left_count > right_count {
                // Most vertices on left - wrap right vertices to left
                if x > 0.5 {
                    -2.0
                } else {
                    0.0
                }
            } else {
                // Most vertices on right - wrap left vertices to right
                if x < -0.5 {
                    2.0
                } else {
                    0.0
                }
            }
        })
        .collect();

    Some(offsets)
}

/// Precomputed map projection data that can be reused across meshes.
///
/// All meshes from the same SphericalVoronoi share the same topology (indices),
/// so projection coordinates and antimeridian wrap offsets can be computed once
/// and reused for all render mode variants.
pub struct MapProjection {
    /// Projected (x, y) coordinates for each vertex.
    pub projected: Vec<(f32, f32)>,
    /// Per-vertex wrap offset for antimeridian handling.
    pub wrap_offset: Vec<f32>,
    /// Cell ranges as (start_tri, end_tri) pairs.
    pub cell_ranges: Vec<(usize, usize)>,
}

impl MapProjection {
    /// Compute map projection data from positions and indices.
    ///
    /// This extracts the expensive geometric computations that are shared
    /// across all meshes with the same topology.
    pub fn compute(positions: &[[f32; 3]], indices: &[u32]) -> Self {
        // First pass: project all vertices
        let projected: Vec<(f32, f32)> = positions
            .iter()
            .map(|p| {
                let pos = Vec3::from_array(*p);
                sphere_to_equirectangular(pos)
            })
            .collect();

        // Second pass: identify cell boundaries from triangle structure
        // Each cell is a triangle fan where all triangles share the same first vertex
        let mut cell_ranges: Vec<(usize, usize)> = Vec::new();
        let mut current_fan_center: Option<u32> = None;
        let mut cell_start = 0;

        for (tri_idx, tri) in indices.chunks(3).enumerate() {
            let fan_center = tri[0];

            match current_fan_center {
                None => {
                    current_fan_center = Some(fan_center);
                }
                Some(prev_center) => {
                    if fan_center != prev_center {
                        cell_ranges.push((cell_start, tri_idx));
                        cell_start = tri_idx;
                        current_fan_center = Some(fan_center);
                    }
                }
            }
        }
        let num_tris = indices.len() / 3;
        if cell_start < num_tris {
            cell_ranges.push((cell_start, num_tris));
        }

        // Third pass: compute wrap offset per cell
        let mut wrap_offset: Vec<f32> = vec![0.0; positions.len()];

        for &(start_tri, end_tri) in &cell_ranges {
            let mut cell_vertex_indices: Vec<usize> = Vec::new();
            for tri_idx in start_tri..end_tri {
                let base = tri_idx * 3;
                cell_vertex_indices.push(indices[base] as usize);
                cell_vertex_indices.push(indices[base + 1] as usize);
                cell_vertex_indices.push(indices[base + 2] as usize);
            }
            cell_vertex_indices.sort();
            cell_vertex_indices.dedup();

            let xs: Vec<f32> = cell_vertex_indices
                .iter()
                .map(|&i| projected[i].0)
                .collect();
            let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Cell crosses antimeridian if it spans more than half the map
            if max_x - min_x > 1.0 {
                let left_count = xs.iter().filter(|&&x| x < 0.0).count();
                let right_count = xs.len() - left_count;

                for &i in &cell_vertex_indices {
                    let x = projected[i].0;
                    if left_count > right_count {
                        if x > 0.5 {
                            wrap_offset[i] = -2.0;
                        }
                    } else if x < -0.5 {
                        wrap_offset[i] = 2.0;
                    }
                }
            }
        }

        Self {
            projected,
            wrap_offset,
            cell_ranges,
        }
    }
}

impl VoronoiMesh {
    /// Generate a mesh with a custom color function per cell.
    pub fn from_voronoi_with_colors<F>(voronoi: &SphericalVoronoi, color_fn: F) -> Self
    where
        F: Fn(usize) -> Vec3,
    {
        Self::from_voronoi_with_elevation(voronoi, color_fn, |_| 0.0)
    }

    /// Generate a mesh with elevation displacement and custom colors.
    /// Vertices are displaced radially: pos * (1.0 + elevation * RELIEF_SCALE)
    /// Vertex elevations are averaged across all cells sharing that vertex to avoid gaps.
    pub fn from_voronoi_with_elevation<C, E>(
        voronoi: &SphericalVoronoi,
        color_fn: C,
        elevation_fn: E,
    ) -> Self
    where
        C: Fn(usize) -> Vec3,
        E: Fn(usize) -> f32,
    {
        use crate::world::RELIEF_SCALE;

        // Step 1: Compute per-vertex elevation by averaging all cells sharing each vertex
        let mut vertex_elevation_sum = vec![0.0f32; voronoi.vertices.len()];
        let mut vertex_cell_count = vec![0u32; voronoi.vertices.len()];

        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            let elevation = elevation_fn(cell_idx);
            for &vertex_idx in cell.vertex_indices {
                let vi = vertex_idx as usize;
                vertex_elevation_sum[vi] += elevation;
                vertex_cell_count[vi] += 1;
            }
        }

        let vertex_elevations: Vec<f32> = vertex_elevation_sum
            .iter()
            .zip(&vertex_cell_count)
            .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
            .collect();

        // Step 2: Build mesh using vertex elevations (not cell elevations)
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut edge_set: HashSet<(u32, u32)> = HashSet::new();

        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            if cell.len() < 3 {
                continue;
            }

            // Color based on cell elevation (not averaged)
            let color = color_fn(cell_idx);

            // Compute wrap offsets for this cell (for map view antimeridian handling)
            let cell_positions: Vec<Vec3> = cell
                .vertex_indices
                .iter()
                .map(|&vi| voronoi.vertices[vi as usize])
                .collect();
            let wrap_offsets = compute_cell_wrap_offsets(&cell_positions);

            // Base index for this cell's vertices
            let base_idx = vertices.len() as u32;

            // Add vertices with displaced positions using averaged elevation
            for (local_idx, &vertex_idx) in cell.vertex_indices.iter().enumerate() {
                let vi = vertex_idx as usize;
                let original_pos = voronoi.vertices[vi];
                let displacement = 1.0 + vertex_elevations[vi] * RELIEF_SCALE;
                let displaced_pos = original_pos * displacement;
                let wrap_offset = wrap_offsets.as_ref().map(|w| w[local_idx]).unwrap_or(0.0);
                vertices.push(
                    MeshVertex::new(displaced_pos, original_pos, color)
                        .with_wrap_offset(wrap_offset),
                );
            }

            // Fan triangulation from first vertex
            let n = cell.vertex_indices.len();
            for i in 1..n - 1 {
                indices.push(base_idx);
                indices.push(base_idx + i as u32);
                indices.push(base_idx + (i + 1) as u32);
            }

            // Collect edges (using global vertex indices from voronoi.vertices)
            for i in 0..n {
                let a = cell.vertex_indices[i] as u32;
                let b = cell.vertex_indices[(i + 1) % n] as u32;
                let edge = if a < b { (a, b) } else { (b, a) };
                edge_set.insert(edge);
            }
        }

        let edge_indices = build_edge_indices(voronoi, &edge_set);

        VoronoiMesh {
            vertices,
            indices,
            edge_indices,
        }
    }

    /// Get edge vertices (separate from fill vertices) for line rendering.
    pub fn edge_vertices(voronoi: &SphericalVoronoi) -> Vec<MeshVertex> {
        let edge_color = Vec3::new(0.35, 0.35, 0.35);
        voronoi
            .vertices
            .iter()
            .map(|&v| MeshVertex::new(v, v, edge_color))
            .collect()
    }

    /// Generate edge line segments with per-edge colors (no elevation displacement).
    /// Returns vertex pairs (non-indexed) for direct line rendering.
    /// `edge_color_fn` takes (vertex_idx_a, vertex_idx_b) and returns the color for that edge.
    pub fn edge_lines_with_colors<F>(
        voronoi: &SphericalVoronoi,
        edge_color_fn: F,
    ) -> Vec<MeshVertex>
    where
        F: Fn(usize, usize) -> Vec3,
    {
        Self::edge_lines_with_elevation(voronoi, edge_color_fn, |_| 0.0, |_| Material::Land)
    }

    /// Generate edge line segments with elevation displacement and per-edge colors.
    /// Uses same water-aware boundary handling as `UnifiedMesh::from_voronoi_with_elevation`.
    pub fn edge_lines_with_elevation<F, E, M>(
        voronoi: &SphericalVoronoi,
        edge_color_fn: F,
        elevation_fn: E,
        material_fn: M,
    ) -> Vec<MeshVertex>
    where
        F: Fn(usize, usize) -> Vec3,
        E: Fn(usize) -> f32,
        M: Fn(usize) -> Material,
    {
        use crate::world::RELIEF_SCALE;

        // Precompute materials and elevations
        let cell_materials: Vec<Material> =
            (0..voronoi.num_cells()).map(|i| material_fn(i)).collect();
        let cell_elevations: Vec<f32> = (0..voronoi.num_cells()).map(|i| elevation_fn(i)).collect();

        // Water-aware vertex elevation (same logic as UnifiedMesh)
        let mut vertex_land_sum = vec![0.0f32; voronoi.vertices.len()];
        let mut vertex_land_count = vec![0u32; voronoi.vertices.len()];
        let mut vertex_water_level = vec![None::<f32>; voronoi.vertices.len()];

        for cell_idx in 0..voronoi.num_cells() {
            let cell = voronoi.cell(cell_idx);
            let material = cell_materials[cell_idx];
            let elevation = cell_elevations[cell_idx];
            let is_water = matches!(material, Material::Ocean | Material::Lake);

            for &vertex_idx in cell.vertex_indices {
                let vi = vertex_idx as usize;
                if is_water {
                    vertex_water_level[vi] = Some(
                        vertex_water_level[vi]
                            .map(|wl| wl.max(elevation))
                            .unwrap_or(elevation),
                    );
                } else {
                    vertex_land_sum[vi] += elevation;
                    vertex_land_count[vi] += 1;
                }
            }
        }

        // Compute final vertex elevations with water boundary handling
        let vertex_elevations: Vec<f32> = (0..voronoi.vertices.len())
            .map(|v| {
                let water_level = vertex_water_level[v];
                let land_count = vertex_land_count[v];

                match (water_level, land_count) {
                    (Some(wl), 0) => wl,
                    (None, n) if n > 0 => vertex_land_sum[v] / n as f32,
                    (Some(wl), _n) => wl, // Land touching water uses water level
                    _ => 0.0,
                }
            })
            .collect();

        let mut vertices = Vec::new();
        let mut processed: HashSet<(u32, u32)> = HashSet::new();

        // Collect all unique edges from all cells
        for cell in voronoi.iter_cells() {
            let n = cell.len();
            for i in 0..n {
                let a = cell.vertex_indices[i];
                let b = cell.vertex_indices[(i + 1) % n];

                let edge = if a < b { (a, b) } else { (b, a) };

                if processed.contains(&edge) {
                    continue;
                }
                processed.insert(edge);

                let v0 = voronoi.vertices[edge.0 as usize];
                let v1 = voronoi.vertices[edge.1 as usize];
                let color = edge_color_fn(edge.0 as usize, edge.1 as usize);

                // Displace by elevation plus slight offset for z-fighting
                let displacement0 = 1.0 + vertex_elevations[edge.0 as usize] * RELIEF_SCALE;
                let displacement1 = 1.0 + vertex_elevations[edge.1 as usize] * RELIEF_SCALE;
                let v0_lifted = v0 * displacement0 * 1.001;
                let v1_lifted = v1 * displacement1 * 1.001;

                vertices.push(MeshVertex::new(v0_lifted, v0, color));
                vertices.push(MeshVertex::new(v1_lifted, v1, color));
            }
        }

        vertices
    }

    /// Project vertices to 2D equirectangular map coordinates.
    /// Returns new vertex buffer with x,y in [-1,1] and z=0.
    /// Handles wrapping at the antimeridian by adjusting per-cell.
    pub fn to_map_vertices(&self) -> (Vec<MeshVertex>, Vec<u32>) {
        // First pass: project all vertices
        let projected: Vec<(f32, f32)> = self
            .vertices
            .iter()
            .map(|v| {
                let p = Vec3::from_array(v.position);
                sphere_to_equirectangular(p)
            })
            .collect();

        // Second pass: identify cell boundaries from triangle structure
        // Each cell is a triangle fan where all triangles share the same first vertex (fan center)
        // Cell boundaries occur when the fan center changes
        let mut cell_ranges: Vec<(usize, usize)> = Vec::new(); // (start_tri, end_tri) exclusive
        let mut current_fan_center: Option<u32> = None;
        let mut cell_start = 0;

        for (tri_idx, tri) in self.indices.chunks(3).enumerate() {
            let fan_center = tri[0];

            match current_fan_center {
                None => {
                    current_fan_center = Some(fan_center);
                }
                Some(prev_center) => {
                    // New cell if fan center changed
                    if fan_center != prev_center {
                        cell_ranges.push((cell_start, tri_idx));
                        cell_start = tri_idx;
                        current_fan_center = Some(fan_center);
                    }
                }
            }
        }
        // Don't forget the last cell
        let num_tris = self.indices.len() / 3;
        if cell_start < num_tris {
            cell_ranges.push((cell_start, num_tris));
        }

        // Third pass: compute wrap offset per cell
        let mut wrap_offset: Vec<f32> = vec![0.0; self.vertices.len()];

        for &(start_tri, end_tri) in &cell_ranges {
            // Collect all vertex indices for this cell
            let mut cell_vertex_indices: Vec<usize> = Vec::new();
            for tri_idx in start_tri..end_tri {
                let base = tri_idx * 3;
                cell_vertex_indices.push(self.indices[base] as usize);
                cell_vertex_indices.push(self.indices[base + 1] as usize);
                cell_vertex_indices.push(self.indices[base + 2] as usize);
            }
            cell_vertex_indices.sort();
            cell_vertex_indices.dedup();

            let xs: Vec<f32> = cell_vertex_indices
                .iter()
                .map(|&i| projected[i].0)
                .collect();
            let min_x = xs.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_x = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Cell crosses antimeridian if it spans more than half the map
            if max_x - min_x > 1.0 {
                let left_count = xs.iter().filter(|&&x| x < 0.0).count();
                let right_count = xs.len() - left_count;

                // Apply consistent wrap to all vertices in this cell
                for &i in &cell_vertex_indices {
                    let x = projected[i].0;
                    if left_count > right_count {
                        // Most on left - wrap right vertices to left
                        if x > 0.5 {
                            wrap_offset[i] = -2.0;
                        }
                    } else {
                        // Most on right - wrap left vertices to right
                        if x < -0.5 {
                            wrap_offset[i] = 2.0;
                        }
                    }
                }
            }
        }

        // Fourth pass: build output with wrapped coordinates
        let mut out_vertices = Vec::new();
        let mut out_indices = Vec::new();

        for tri in self.indices.chunks(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let base = out_vertices.len() as u32;

            for &i in &[i0, i1, i2] {
                let (x, y) = projected[i];
                out_vertices.push(MeshVertex {
                    position: [x + wrap_offset[i], y, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    color: self.vertices[i].color,
                    wrap_offset: 0.0, // Not used in 2D map
                });
            }

            out_indices.push(base);
            out_indices.push(base + 1);
            out_indices.push(base + 2);
        }

        (out_vertices, out_indices)
    }

    /// Project vertices using precomputed projection data.
    ///
    /// This is much faster than `to_map_vertices` when projecting multiple
    /// meshes with the same topology, as the expensive cell boundary detection
    /// and wrap offset computation is done once and reused.
    pub fn to_map_vertices_with_projection(
        &self,
        projection: &MapProjection,
    ) -> (Vec<MeshVertex>, Vec<u32>) {
        let mut out_vertices = Vec::with_capacity(self.indices.len());
        let mut out_indices = Vec::with_capacity(self.indices.len());

        for tri in self.indices.chunks(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let base = out_vertices.len() as u32;

            for &i in &[i0, i1, i2] {
                let (x, y) = projection.projected[i];
                out_vertices.push(MeshVertex {
                    position: [x + projection.wrap_offset[i], y, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    color: self.vertices[i].color,
                    wrap_offset: 0.0, // Not used in 2D map
                });
            }

            out_indices.push(base);
            out_indices.push(base + 1);
            out_indices.push(base + 2);
        }

        (out_vertices, out_indices)
    }

    /// Compute map projection data from this mesh.
    ///
    /// The returned projection can be reused with other meshes that have
    /// the same topology (same indices).
    pub fn compute_projection(&self) -> MapProjection {
        let positions: Vec<[f32; 3]> = self.vertices.iter().map(|v| v.position).collect();
        MapProjection::compute(&positions, &self.indices)
    }

    /// Project edge vertices to 2D equirectangular map coordinates.
    /// Returns vertices and indices, handling antimeridian wrapping per-edge.
    pub fn to_map_edge_vertices(&self, voronoi: &SphericalVoronoi) -> (Vec<MeshVertex>, Vec<u32>) {
        let edge_color = Vec3::new(0.35, 0.35, 0.35);
        let mut out_vertices = Vec::new();
        let mut out_indices = Vec::new();

        // Process edges in pairs
        for edge in self.edge_indices.chunks(2) {
            if edge.len() < 2 {
                continue;
            }
            let i0 = edge[0] as usize;
            let i1 = edge[1] as usize;

            let v0 = voronoi.vertices[i0];
            let v1 = voronoi.vertices[i1];

            let (mut x0, y0) = sphere_to_equirectangular(v0);
            let (mut x1, y1) = sphere_to_equirectangular(v1);

            // Check if edge crosses antimeridian
            let span = (x0 - x1).abs();
            if span > 1.0 {
                // Edge crosses antimeridian - wrap to same side
                if x0 < 0.0 && x1 > 0.0 {
                    if x0 < -0.5 {
                        x0 += 2.0;
                    } else {
                        x1 -= 2.0;
                    }
                } else if x0 > 0.0 && x1 < 0.0 {
                    if x1 < -0.5 {
                        x1 += 2.0;
                    } else {
                        x0 -= 2.0;
                    }
                }
            }

            let base = out_vertices.len() as u32;
            out_vertices.push(MeshVertex::new(
                Vec3::new(x0, y0, 0.001),
                Vec3::Z,
                edge_color,
            ));
            out_vertices.push(MeshVertex::new(
                Vec3::new(x1, y1, 0.001),
                Vec3::Z,
                edge_color,
            ));
            out_indices.push(base);
            out_indices.push(base + 1);
        }

        (out_vertices, out_indices)
    }
}

/// Build edge indices from the edge set.
/// Returns indices into the voronoi.vertices array.
fn build_edge_indices(_voronoi: &SphericalVoronoi, edge_set: &HashSet<(u32, u32)>) -> Vec<u32> {
    edge_set.iter().flat_map(|&(a, b)| [a, b]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{lloyd_relax, random_sphere_points, SphericalVoronoi};

    #[test]
    fn test_mesh_generation() {
        let mut points = random_sphere_points(50);
        lloyd_relax(&mut points, 10);
        let voronoi = SphericalVoronoi::compute(&points);
        let mesh = VoronoiMesh::from_voronoi_with_colors(&voronoi, |_| Vec3::new(0.5, 0.5, 0.5));

        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert!(!mesh.edge_indices.is_empty());

        // Indices should be valid
        let max_idx = mesh.vertices.len() as u32;
        for &idx in &mesh.indices {
            assert!(idx < max_idx, "Invalid triangle index: {}", idx);
        }
    }
}
