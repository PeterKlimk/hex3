//! Cell building types and algorithms for spherical Voronoi computation.

use glam::Vec3;

/// Vertex data: (triplet key, position). Uses u32 indices to save space.
pub type VertexData = ([u32; 3], Vec3);
/// Vertex list for a single cell.
pub type VertexList = Vec<VertexData>;

// Epsilon values for numerical stability
pub(crate) const EPS_PLANE_CONTAINS: f32 = 1e-7;
pub(crate) const EPS_PLANE_CLIP: f32 = 1e-7;
pub(crate) const EPS_PLANE_PARALLEL: f32 = 1e-6;
pub(crate) const EPS_TERMINATION_MARGIN: f32 = 1e-7;

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
    /// True once a seed polygon has been initialized.
    seeded: bool,
    /// True if clipping has already eliminated the cell (intersection empty).
    dead: bool,
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
            seeded: false,
            dead: false,
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
        self.seeded = false;
        self.dead = false;
    }

    /// Returns true if this neighbor has already been clipped into the cell.
    #[inline]
    pub fn has_neighbor(&self, neighbor_idx: usize) -> bool {
        self.neighbor_indices.contains(&neighbor_idx)
    }

    /// Returns an iterator over the neighbor indices that have been clipped.
    /// Useful for building a HashSet for O(1) lookup during full-scan fallback.
    #[inline]
    pub fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbor_indices.iter().copied()
    }

    #[inline]
    fn intersect_planes_in_triplet(
        &self,
        plane_a: usize,
        plane_b: usize,
        plane_c: usize,
    ) -> Option<Vec3> {
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
            if self.planes[plane_c].signed_distance(v) >= -EPS_PLANE_CLIP {
                return Some(v);
            }
        }
        None
    }

    /// Compute exact intersection of two great circle planes on unit sphere.
    /// Returns the candidate point closer to `hint` (typically arc midpoint or generator).
    #[inline]
    fn intersect_two_planes(plane_a: &GreatCircle, plane_b: &GreatCircle, hint: Vec3) -> Vec3 {
        let cross = plane_a.normal.cross(plane_b.normal);
        let len_sq = cross.length_squared();
        if len_sq < EPS_PLANE_PARALLEL * EPS_PLANE_PARALLEL {
            // Parallel planes - fall back to hint direction
            return hint.normalize();
        }
        let v = cross / len_sq.sqrt();
        // Pick the candidate closer to hint
        if v.dot(hint) >= 0.0 { v } else { -v }
    }

    fn seed_from_triplet(&mut self, a: usize, b: usize, c: usize) -> bool {
        let v0 = match self.intersect_planes_in_triplet(a, b, c) {
            Some(v) => v,
            None => return false,
        };
        let v1 = match self.intersect_planes_in_triplet(b, c, a) {
            Some(v) => v,
            None => return false,
        };
        let v2 = match self.intersect_planes_in_triplet(c, a, b) {
            Some(v) => v,
            None => return false,
        };

        // CRITICAL FIX: Check that all 3 vertices satisfy ALL accumulated half-spaces,
        // not just the 3 in the triplet. Otherwise the seed triangle may be entirely
        // outside some other plane, causing the cell to go dead immediately.
        for plane_idx in 0..self.planes.len() {
            if plane_idx == a || plane_idx == b || plane_idx == c {
                continue;
            }
            let plane = &self.planes[plane_idx];
            // If all 3 vertices are outside this plane, reject the triplet
            let d0 = plane.signed_distance(v0);
            let d1 = plane.signed_distance(v1);
            let d2 = plane.signed_distance(v2);
            if d0 < -EPS_PLANE_CLIP && d1 < -EPS_PLANE_CLIP && d2 < -EPS_PLANE_CLIP {
                return false;
            }
        }

        // Compute triangle normal and check winding
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2);

        // CRITICAL: Verify the generator is inside the spherical triangle.
        // A valid seed triangle must contain the generator; otherwise later clipping
        // can eliminate the entire polygon since it doesn't contain the fixed point.
        // For a spherical triangle, the generator is inside if it's on the positive
        // side of all 3 edge planes (great circles through each pair of vertices).
        let edge_plane_01 = v0.cross(v1); // normal to great circle through v0, v1
        let edge_plane_12 = v1.cross(v2);
        let edge_plane_20 = v2.cross(v0);

        // Check winding direction and generator containment together
        let g = self.generator;
        let inside_01 = edge_plane_01.dot(g);
        let inside_12 = edge_plane_12.dot(g);
        let inside_20 = edge_plane_20.dot(g);

        // All should have the same sign for generator to be inside
        let consistent_winding = (inside_01 > 0.0 && inside_12 > 0.0 && inside_20 > 0.0)
            || (inside_01 < 0.0 && inside_12 < 0.0 && inside_20 < 0.0);

        if !consistent_winding {
            return false;
        }

        self.vertices.clear();
        self.edge_planes.clear();

        if normal.dot(self.generator) >= 0.0 {
            self.vertices.push(CellVertex { pos: v0, plane_a: a, plane_b: b });
            self.vertices.push(CellVertex { pos: v1, plane_a: b, plane_b: c });
            self.vertices.push(CellVertex { pos: v2, plane_a: c, plane_b: a });
            self.edge_planes.push(b);
            self.edge_planes.push(c);
            self.edge_planes.push(a);
        } else {
            self.vertices.push(CellVertex { pos: v0, plane_a: a, plane_b: b });
            self.vertices.push(CellVertex { pos: v2, plane_a: c, plane_b: a });
            self.vertices.push(CellVertex { pos: v1, plane_a: b, plane_b: c });
            self.edge_planes.push(a);
            self.edge_planes.push(c);
            self.edge_planes.push(b);
        }

        true
    }

    fn try_seed_from_plane(&mut self, new_plane_idx: usize) -> Option<[usize; 3]> {
        // First priority: try triplets including the newest plane (most likely to succeed)
        for i in 0..new_plane_idx {
            for j in (i + 1)..new_plane_idx {
                if self.seed_from_triplet(i, j, new_plane_idx) {
                    return Some([i, j, new_plane_idx]);
                }
            }
        }

        // Second priority: try ALL triplets among accumulated planes.
        // This is needed because earlier triplets may have failed only because
        // their vertices were outside planes that hadn't been added yet.
        // Now with more planes, the full half-space check in seed_from_triplet
        // can properly reject bad triplets and find good ones.
        for a in 0..new_plane_idx {
            for b in (a + 1)..new_plane_idx {
                for c in (b + 1)..new_plane_idx {
                    if self.seed_from_triplet(a, b, c) {
                        return Some([a, b, c]);
                    }
                }
            }
        }

        None
    }

    fn clip_with_plane(&mut self, plane_idx: usize) {
        let new_plane = self.planes[plane_idx];
        let new_plane_idx = plane_idx;

        let n = self.vertices.len();
        if n < 3 {
            return;
        }

        let mut inside = [false; MAX_VERTICES];
        for (i, v) in self.vertices.iter().enumerate() {
            let dist = new_plane.signed_distance(v.pos);
            inside[i] = dist >= -EPS_PLANE_CLIP;
        }

        let inside_count = inside[..n].iter().filter(|&&x| x).count();
        if inside_count == n {
            return;
        }
        if inside_count == 0 {
            self.vertices.clear();
            self.edge_planes.clear();
            self.dead = true;
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
        // Use interpolation to find approximate point, then exact intersection for precise position
        let d_entry_start = new_plane.signed_distance(entry_start);
        let d_entry_end = new_plane.signed_distance(entry_end);
        let t_entry = d_entry_start / (d_entry_start - d_entry_end);
        let entry_hint = (entry_start * (1.0 - t_entry) + entry_end * t_entry).normalize();
        let entry_vertex = CellVertex {
            pos: Self::intersect_two_planes(&self.planes[entry_edge_plane], &new_plane, entry_hint),
            plane_a: entry_edge_plane,
            plane_b: new_plane_idx,
        };

        let exit_edge_plane = self.edge_planes[exit_idx];
        let exit_start = self.vertices[exit_idx].pos;
        let exit_end = self.vertices[(exit_idx + 1) % n].pos;
        let d_exit_start = new_plane.signed_distance(exit_start);
        let d_exit_end = new_plane.signed_distance(exit_end);
        let t_exit = d_exit_start / (d_exit_start - d_exit_end);
        let exit_hint = (exit_start * (1.0 - t_exit) + exit_end * t_exit).normalize();
        let exit_vertex = CellVertex {
            pos: Self::intersect_two_planes(&self.planes[exit_edge_plane], &new_plane, exit_hint),
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

    /// Add a new plane (neighbor's bisector) and clip the cell. O(n).
    /// Skips neighbors that are too close to the generator (degenerate bisector).
    pub fn clip(&mut self, neighbor_idx: usize, neighbor: Vec3) {
        // Skip neighbors that are too close - their bisector is numerically unstable
        let diff = self.generator - neighbor;
        if diff.length_squared() < super::MIN_BISECTOR_DISTANCE * super::MIN_BISECTOR_DISTANCE {
            return;
        }

        let new_plane = GreatCircle::bisector(self.generator, neighbor);
        let new_plane_idx = self.planes.len();
        self.planes.push(new_plane);
        self.neighbor_indices.push(neighbor_idx);

        if !self.seeded {
            if self.planes.len() < 3 {
                return;
            }
            let seed = match self.try_seed_from_plane(new_plane_idx) {
                Some(seed) => seed,
                None => return,
            };
            self.seeded = true;
            for plane_idx in 0..self.planes.len() {
                if plane_idx == seed[0] || plane_idx == seed[1] || plane_idx == seed[2] {
                    continue;
                }
                self.clip_with_plane(plane_idx);
                if self.dead {
                    return;
                }
            }
            return;
        }

        self.clip_with_plane(new_plane_idx);
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

    /// Attempt to reseed using the best-conditioned triplet among all planes.
    /// Returns true if a non-degenerate cell is recovered.
    pub fn try_reseed_best(&mut self) -> bool {
        let plane_count = self.planes.len();
        if plane_count < 3 {
            return false;
        }

        let mut best_triplet: Option<(usize, usize, usize)> = None;
        let mut best_score = 0.0f32;
        let min_score = EPS_PLANE_PARALLEL * EPS_PLANE_PARALLEL;

        for a in 0..plane_count {
            for b in (a + 1)..plane_count {
                for c in (b + 1)..plane_count {
                    let na = self.planes[a].normal;
                    let nb = self.planes[b].normal;
                    let nc = self.planes[c].normal;
                    let ab = na.cross(nb).length_squared();
                    let bc = nb.cross(nc).length_squared();
                    let ca = nc.cross(na).length_squared();
                    let score = ab.min(bc).min(ca);
                    if score <= best_score.max(min_score) {
                        continue;
                    }

                    self.vertices.clear();
                    self.edge_planes.clear();
                    self.dead = false;
                    self.seeded = false;

                    if !self.seed_from_triplet(a, b, c) {
                        continue;
                    }

                    best_score = score;
                    best_triplet = Some((a, b, c));
                }
            }
        }

        let (a, b, c) = match best_triplet {
            Some(t) => t,
            None => return false,
        };

        self.vertices.clear();
        self.edge_planes.clear();
        self.dead = false;
        self.seeded = false;
        if !self.seed_from_triplet(a, b, c) {
            return false;
        }
        self.seeded = true;

        for plane_idx in 0..plane_count {
            if plane_idx == a || plane_idx == b || plane_idx == c {
                continue;
            }
            self.clip_with_plane(plane_idx);
            if self.dead {
                return false;
            }
        }

        self.vertices.len() >= 3
    }

    /// Get the final vertices with their canonical triplet keys.
    pub fn get_vertices_with_keys(&self) -> VertexList {
        let mut out = Vec::with_capacity(self.vertices.len());
        self.get_vertices_into(&mut out);
        out
    }

    /// Write vertices with canonical triplet keys into the provided buffer.
    /// Returns the number of vertices written.
    #[inline]
    pub fn get_vertices_into(&self, out: &mut Vec<VertexData>) -> usize {
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

    #[inline]
    pub fn planes_count(&self) -> usize {
        self.planes.len()
    }

    #[inline]
    pub fn is_dead(&self) -> bool {
        self.dead
    }

    #[inline]
    pub fn is_seeded(&self) -> bool {
        self.seeded
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
