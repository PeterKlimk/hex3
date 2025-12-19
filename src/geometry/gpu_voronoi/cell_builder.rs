//! Cell building types and algorithms for spherical Voronoi computation.

use glam::Vec3;

/// Vertex data: (triplet key, position). Uses u32 indices to save space.
pub type VertexData = ([u32; 3], Vec3);
/// Vertex list for a single cell.
pub type VertexList = Vec<VertexData>;

// Epsilon values for numerical stability
pub(crate) const EPS_PLANE_CONTAINS: f32 = 1e-10;
pub(crate) const EPS_PLANE_CLIP: f32 = 1e-9;
pub(crate) const EPS_PLANE_PARALLEL: f32 = 1e-10;
pub(crate) const EPS_TERMINATION_MARGIN: f32 = 1e-6;

/// Maximum number of neighbors to consider per cell.
pub const DEFAULT_K: usize = 24;

/// Maximum number of planes (great circle boundaries) per cell.
pub const MAX_PLANES: usize = 24;

/// Maximum number of vertices (plane triplet intersections) per cell.
pub const MAX_VERTICES: usize = 32;

/// A great circle on the unit sphere, represented by its normal vector.
#[derive(Debug, Clone, Copy)]
pub struct GreatCircle {
    pub normal: Vec3,
}

impl GreatCircle {
    /// Create the bisector great circle between two points on the unit sphere.
    #[inline]
    pub fn bisector(a: Vec3, b: Vec3) -> Self {
        let normal = (a - b).normalize();
        GreatCircle { normal }
    }

    /// Check if a point is in the positive hemisphere.
    #[inline]
    pub fn contains(&self, point: Vec3) -> bool {
        self.normal.dot(point) >= -EPS_PLANE_CONTAINS
    }

    /// Signed distance from point to the great circle plane.
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point)
    }
}

/// A vertex in the incrementally-built cell polygon.
#[derive(Debug, Clone, Copy)]
struct CellVertex {
    pos: Vec3,
    plane_a: usize,
    plane_b: usize,
}

/// Epsilon for detecting degenerate vertices (4+ equidistant generators).
/// Set to match the merge tolerance in dedup.rs to avoid missing cases.
const EPS_DEGENERACY: f32 = 1e-5;

/// Incremental cell builder using ordered polygon clipping.
/// O(n) per plane addition instead of O(n³) brute-force.
#[derive(Debug, Clone)]
pub struct IncrementalCellBuilder {
    generator_idx: usize,
    generator: Vec3,
    planes: Vec<GreatCircle>,
    neighbor_indices: Vec<usize>,
    vertices: Vec<CellVertex>,
    edge_planes: Vec<usize>,
    tmp_vertices: Vec<CellVertex>,
    tmp_edge_planes: Vec<usize>,
    /// True if any vertex lies exactly on a clipping plane (4+ equidistant generators).
    has_degeneracy: bool,
    /// Triplet equivalence edges implied by degeneracies (for union-find unification).
    /// Each edge indicates two distinct triplets that should refer to the same vertex.
    degenerate_edges: Vec<([u32; 3], [u32; 3])>,
}

impl IncrementalCellBuilder {
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        Self {
            generator_idx,
            generator,
            planes: Vec::with_capacity(MAX_PLANES),
            neighbor_indices: Vec::with_capacity(MAX_PLANES),
            vertices: Vec::with_capacity(MAX_VERTICES),
            edge_planes: Vec::with_capacity(MAX_VERTICES),
            tmp_vertices: Vec::with_capacity(MAX_VERTICES),
            tmp_edge_planes: Vec::with_capacity(MAX_VERTICES),
            has_degeneracy: false,
            degenerate_edges: Vec::new(),
        }
    }

    pub fn reset(&mut self, generator_idx: usize, generator: Vec3) {
        self.generator_idx = generator_idx;
        self.generator = generator;
        self.planes.clear();
        self.neighbor_indices.clear();
        self.vertices.clear();
        self.edge_planes.clear();
        self.tmp_vertices.clear();
        self.tmp_edge_planes.clear();
        self.has_degeneracy = false;
        self.degenerate_edges.clear();
    }

    /// Returns true if a degeneracy was detected (vertex exactly on a clipping plane).
    #[inline]
    pub fn has_degeneracy(&self) -> bool {
        self.has_degeneracy
    }

    /// Returns the triplet equivalence edges implied by degeneracies.
    #[inline]
    pub fn degenerate_edges(&self) -> &[([u32; 3], [u32; 3])] {
        &self.degenerate_edges
    }

    /// Create a sorted triplet key from three generator indices.
    #[inline]
    fn make_triplet(&self, a: usize, b: usize) -> [u32; 3] {
        let mut t = [
            self.generator_idx as u32,
            self.neighbor_indices[a] as u32,
            self.neighbor_indices[b] as u32,
        ];
        if t[0] > t[1] { t.swap(0, 1); }
        if t[1] > t[2] { t.swap(1, 2); }
        if t[0] > t[1] { t.swap(0, 1); }
        t
    }

    #[inline]
    fn intersect_planes_valid(&self, plane_a: usize, plane_b: usize) -> Option<Vec3> {
        let n_a = self.planes[plane_a].normal;
        let n_b = self.planes[plane_b].normal;
        let cross = n_a.cross(n_b);
        let len = cross.length();
        if len < EPS_PLANE_PARALLEL {
            return None;
        }
        let v1 = cross / len;
        let v2 = -v1;

        let (primary, secondary) = if v1.dot(self.generator) >= v2.dot(self.generator) {
            (v1, v2)
        } else {
            (v2, v1)
        };

        for v in [primary, secondary] {
            let inside_all = self.planes.iter().enumerate().all(|(k, plane)| {
                k == plane_a || k == plane_b || plane.signed_distance(v) >= -EPS_PLANE_CLIP
            });
            if inside_all {
                return Some(v);
            }
        }
        None
    }

    #[inline]
    fn intersect_arc_with_plane(arc_start: Vec3, arc_end: Vec3, plane: &GreatCircle) -> Vec3 {
        let d_start = plane.signed_distance(arc_start);
        let d_end = plane.signed_distance(arc_end);
        let t = d_start / (d_start - d_end);
        let p = arc_start * (1.0 - t) + arc_end * t;
        p.normalize()
    }

    /// Add a new plane (neighbor's bisector) and clip the cell. O(n).
    pub fn clip(&mut self, neighbor_idx: usize, neighbor: Vec3) {
        let new_plane = GreatCircle::bisector(self.generator, neighbor);
        let new_plane_idx = self.planes.len();
        self.planes.push(new_plane);
        self.neighbor_indices.push(neighbor_idx);

        if self.planes.len() < 3 {
            return;
        }
        if self.planes.len() == 3 {
            let v0 = match self.intersect_planes_valid(0, 1) {
                Some(v) => v,
                None => return,
            };
            let v1 = match self.intersect_planes_valid(1, 2) {
                Some(v) => v,
                None => return,
            };
            let v2 = match self.intersect_planes_valid(2, 0) {
                Some(v) => v,
                None => return,
            };

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = edge1.cross(edge2);

            if normal.dot(self.generator) >= 0.0 {
                self.vertices.clear();
                self.vertices.push(CellVertex { pos: v0, plane_a: 0, plane_b: 1 });
                self.vertices.push(CellVertex { pos: v1, plane_a: 1, plane_b: 2 });
                self.vertices.push(CellVertex { pos: v2, plane_a: 2, plane_b: 0 });
                self.edge_planes.clear();
                self.edge_planes.push(1);
                self.edge_planes.push(2);
                self.edge_planes.push(0);
            } else {
                self.vertices.clear();
                self.vertices.push(CellVertex { pos: v0, plane_a: 0, plane_b: 1 });
                self.vertices.push(CellVertex { pos: v2, plane_a: 2, plane_b: 0 });
                self.vertices.push(CellVertex { pos: v1, plane_a: 1, plane_b: 2 });
                self.edge_planes.clear();
                self.edge_planes.push(0);
                self.edge_planes.push(2);
                self.edge_planes.push(1);
            }
            return;
        }

        let n = self.vertices.len();
        if n < 3 {
            return;
        }

        let mut inside = [false; MAX_VERTICES];
        for (i, v) in self.vertices.iter().enumerate() {
            let dist = new_plane.signed_distance(v.pos);
            inside[i] = dist >= -EPS_PLANE_CLIP;
            // Detect degeneracy: vertex lies exactly on the new plane (4+ equidistant generators)
            if dist.abs() < EPS_DEGENERACY {
                self.has_degeneracy = true;
                // Record triplet equivalences implied by 4+ equidistant generators.
                // The same geometric vertex can be represented by multiple plane pairs
                // depending on which 3-of-4 generators define it.
                let t_ab = self.make_triplet(v.plane_a, v.plane_b);
                let t_a_new = self.make_triplet(v.plane_a, new_plane_idx);
                let t_b_new = self.make_triplet(v.plane_b, new_plane_idx);
                self.degenerate_edges.push((t_ab, t_a_new));
                self.degenerate_edges.push((t_ab, t_b_new));
            }
        }

        let inside_count = inside[..n].iter().filter(|&&x| x).count();
        if inside_count == n {
            return;
        }
        if inside_count == 0 {
            self.vertices.clear();
            self.edge_planes.clear();
            return;
        }

        let mut entry_idx: Option<usize> = None;
        let mut exit_idx: Option<usize> = None;

        for i in 0..n {
            let next = (i + 1) % n;
            if inside[i] && !inside[next] {
                exit_idx = Some(i);
            }
            if !inside[i] && inside[next] {
                entry_idx = Some(i);
            }
        }

        let (entry_idx, exit_idx) = match (entry_idx, exit_idx) {
            (Some(e), Some(x)) => (e, x),
            _ => return,
        };

        let entry_edge_plane = self.edge_planes[entry_idx];
        let entry_start = self.vertices[entry_idx].pos;
        let entry_end = self.vertices[(entry_idx + 1) % n].pos;
        let entry_vertex = CellVertex {
            pos: Self::intersect_arc_with_plane(entry_start, entry_end, &new_plane),
            plane_a: entry_edge_plane,
            plane_b: new_plane_idx,
        };

        let exit_edge_plane = self.edge_planes[exit_idx];
        let exit_start = self.vertices[exit_idx].pos;
        let exit_end = self.vertices[(exit_idx + 1) % n].pos;
        let exit_vertex = CellVertex {
            pos: Self::intersect_arc_with_plane(exit_start, exit_end, &new_plane),
            plane_a: exit_edge_plane,
            plane_b: new_plane_idx,
        };

        self.tmp_vertices.clear();
        self.tmp_edge_planes.clear();
        self.tmp_vertices.reserve(n);
        self.tmp_edge_planes.reserve(n);

        self.tmp_vertices.push(entry_vertex);
        self.tmp_edge_planes.push(entry_edge_plane);

        let mut i = (entry_idx + 1) % n;
        while i != (exit_idx + 1) % n {
            self.tmp_vertices.push(self.vertices[i]);
            self.tmp_edge_planes.push(self.edge_planes[i]);
            i = (i + 1) % n;
        }

        self.tmp_vertices.push(exit_vertex);
        self.tmp_edge_planes.push(new_plane_idx);

        std::mem::swap(&mut self.vertices, &mut self.tmp_vertices);
        std::mem::swap(&mut self.edge_planes, &mut self.tmp_edge_planes);
    }

    /// Check if we can terminate early based on security radius.
    pub fn can_terminate(&self, next_neighbor_cos: f32) -> bool {
        if self.vertices.len() < 3 {
            return false;
        }

        let min_cos = self.vertices.iter()
            .map(|v| self.generator.dot(v.pos).clamp(-1.0, 1.0))
            .fold(1.0f32, f32::min);

        if min_cos <= 0.0 {
            return false;
        }

        let cos_2max = 2.0 * min_cos * min_cos - 1.0;
        next_neighbor_cos < cos_2max - EPS_TERMINATION_MARGIN
    }

    /// Get the final vertices with their canonical triplet keys.
<<<<<<< HEAD
    pub fn get_vertices_with_keys(&self) -> VertexList {
        let mut out = Vec::with_capacity(self.vertices.len());
        self.get_vertices_into(&mut out);
        out
    }

    /// Write vertices with canonical triplet keys into the provided buffer.
    /// Returns the number of vertices written.
    #[inline]
    pub fn get_vertices_into(&self, out: &mut Vec<VertexData>) -> usize {
=======
    pub fn get_vertices_with_keys(&self) -> super::CellVerts {
>>>>>>> b59c62061cbe57e5edec56153566640a3970d715
        #[inline]
        fn sort3(mut a: [u32; 3]) -> [u32; 3] {
            if a[0] > a[1] { a.swap(0, 1); }
            if a[1] > a[2] { a.swap(1, 2); }
            if a[0] > a[1] { a.swap(0, 1); }
            a
        }

        out.reserve(self.vertices.len());
        let count = self.vertices.len();
        for v in &self.vertices {
            let triplet = sort3([
                self.generator_idx as u32,
                self.neighbor_indices[v.plane_a] as u32,
                self.neighbor_indices[v.plane_b] as u32,
            ]);
            out.push((triplet, v.pos));
        }
        count
    }

    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Write vertices with keys to a provided buffer, returning the slice written.
    /// This avoids per-cell allocation by reusing a thread-local buffer.
    pub fn write_vertices_to_buffer<'a>(
        &self,
        buffer: &'a mut Vec<([usize; 3], Vec3)>,
    ) -> &'a [([usize; 3], Vec3)] {
        #[inline]
        fn sort3(mut a: [usize; 3]) -> [usize; 3] {
            if a[0] > a[1] { a.swap(0, 1); }
            if a[1] > a[2] { a.swap(1, 2); }
            if a[0] > a[1] { a.swap(0, 1); }
            a
        }

        let start = buffer.len();
        for v in &self.vertices {
            let triplet = sort3([
                self.generator_idx,
                self.neighbor_indices[v.plane_a],
                self.neighbor_indices[v.plane_b],
            ]);
            buffer.push((triplet, v.pos));
        }
        &buffer[start..]
    }
}

/// Compute geodesic distance between two points on unit sphere.
#[inline]
pub fn geodesic_distance(a: Vec3, b: Vec3) -> f32 {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    dot.acos()
}

/// Order vertices counter-clockwise around a generator when viewed from outside.
pub fn order_vertices_ccw_indices(generator: Vec3, vertices: &[Vec3]) -> Vec<usize> {
    if vertices.len() <= 2 {
        return (0..vertices.len()).collect();
    }

    let up = if generator.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent_x = generator.cross(up).normalize();
    let tangent_y = generator.cross(tangent_x).normalize();

    let mut indexed: Vec<(usize, f32)> = vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let to_point = v - generator * generator.dot(v);
            let x = to_point.dot(tangent_x);
            let y = to_point.dot(tangent_y);
            (i, y.atan2(x))
        })
        .collect();

    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));
    indexed.into_iter().map(|(i, _)| i).collect()
}
