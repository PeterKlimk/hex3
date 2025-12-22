//! Cell building types and algorithms for spherical Voronoi computation.

use glam::{DVec3, Vec3};

use super::constants::{
    support_cluster_drift_dot, EPS_PLANE_CLIP, EPS_PLANE_CONDITION, EPS_PLANE_CONTAINS,
    EPS_PLANE_PARALLEL, EPS_TERMINATION_MARGIN, MIN_BISECTOR_DISTANCE, SUPPORT_CERT_MARGIN_ABS,
    SUPPORT_CLUSTER_RADIUS_ANGLE, SUPPORT_EPS_ABS, SUPPORT_VERTEX_ANGLE_EPS,
};

/// Vertex key for deduplication (fast triplet or full support set).
#[derive(Debug, Clone, Copy)]
pub enum VertexKey {
    Triplet([u32; 3]),
    Support { start: u32, len: u8 },
}

/// Vertex data: (key, position). Uses u32 indices to save space.
pub type VertexData = (VertexKey, Vec3);
/// Vertex list for a single cell.
pub type VertexList = Vec<VertexData>;

// Epsilon values are defined in constants.rs.

pub struct GapSampler {
    count: usize,
    min: f64,
    sample: Vec<f64>,
    limit: usize,
    rng: u64,
}

impl GapSampler {
    pub fn new(limit: usize, seed: u64) -> Self {
        let seed = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self {
            count: 0,
            min: f64::INFINITY,
            sample: Vec::with_capacity(limit),
            limit,
            rng: seed,
        }
    }

    #[inline]
    pub(crate) fn record(&mut self, gap: f64) {
        if !gap.is_finite() {
            return;
        }
        self.count += 1;
        if gap < self.min {
            self.min = gap;
        }
        if self.limit == 0 {
            return;
        }
        if self.sample.len() < self.limit {
            self.sample.push(gap);
            return;
        }
        let idx = (self.next_u64() % self.count as u64) as usize;
        if idx < self.limit {
            self.sample[idx] = gap;
        }
    }

    #[inline]
    pub(crate) fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub(crate) fn min(&self) -> f64 {
        self.min
    }

    #[inline]
    pub(crate) fn sample(&self) -> &[f64] {
        &self.sample
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // xorshift64
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }
}

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
    ill_conditioned: bool,
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
    cached_min_vertex_cos: f32,
    cached_min_vertex_cos_valid: bool,
    sin_eps: f32,
    cos_eps: f32,
    termination_margin: f32,
}

impl IncrementalCellBuilder {
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        let eps_cell = SUPPORT_VERTEX_ANGLE_EPS + SUPPORT_CLUSTER_RADIUS_ANGLE;
        let (sin_eps, cos_eps) = (eps_cell as f32).sin_cos();
        let termination_margin = EPS_TERMINATION_MARGIN
            + SUPPORT_CERT_MARGIN_ABS as f32
            + 2.0 * SUPPORT_EPS_ABS as f32
            + 2.0 * support_cluster_drift_dot() as f32;
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
            cached_min_vertex_cos: 1.0,
            cached_min_vertex_cos_valid: false,
            sin_eps,
            cos_eps,
            termination_margin,
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
        self.cached_min_vertex_cos = 1.0;
        self.cached_min_vertex_cos_valid = false;
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
    ) -> Option<(Vec3, bool)> {
        let n_a = self.planes[plane_a].normal;
        let n_b = self.planes[plane_b].normal;
        let cross = n_a.cross(n_b);
        let len = cross.length();
        if len < EPS_PLANE_PARALLEL {
            return None;
        }
        let ill_conditioned = len < EPS_PLANE_CONDITION;
        let v1 = cross / len;
        let v2 = -v1;

        let (primary, secondary) = if v1.dot(self.generator) >= v2.dot(self.generator) {
            (v1, v2)
        } else {
            (v2, v1)
        };

        for v in [primary, secondary] {
            if self.planes[plane_c].signed_distance(v) >= -EPS_PLANE_CLIP {
                return Some((v, ill_conditioned));
            }
        }
        None
    }

    /// Compute exact intersection of two great circle planes on unit sphere.
    /// Returns the candidate point closer to `hint` (typically arc midpoint or generator).
    #[inline]
    fn intersect_two_planes(
        plane_a: &GreatCircle,
        plane_b: &GreatCircle,
        hint: Vec3,
    ) -> (Vec3, bool) {
        let cross = plane_a.normal.cross(plane_b.normal);
        let len_sq = cross.length_squared();
        if len_sq < EPS_PLANE_PARALLEL * EPS_PLANE_PARALLEL {
            // Parallel planes - fall back to hint direction
            return (hint.normalize(), true);
        }
        let len = len_sq.sqrt();
        let v = cross / len;
        let ill_conditioned = len < EPS_PLANE_CONDITION;
        // Pick the candidate closer to hint
        let v = if v.dot(hint) >= 0.0 { v } else { -v };
        (v, ill_conditioned)
    }

    fn seed_from_triplet(&mut self, a: usize, b: usize, c: usize) -> bool {
        let (v0, v0_ill) = match self.intersect_planes_in_triplet(a, b, c) {
            Some(v) => v,
            None => return false,
        };
        let (v1, v1_ill) = match self.intersect_planes_in_triplet(b, c, a) {
            Some(v) => v,
            None => return false,
        };
        let (v2, v2_ill) = match self.intersect_planes_in_triplet(c, a, b) {
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

        let mut min_cos = 1.0f32;
        if normal.dot(self.generator) >= 0.0 {
            self.vertices.push(CellVertex {
                pos: v0,
                plane_a: a,
                plane_b: b,
                ill_conditioned: v0_ill,
            });
            min_cos = min_cos.min(self.generator.dot(v0).clamp(-1.0, 1.0));
            self.vertices.push(CellVertex {
                pos: v1,
                plane_a: b,
                plane_b: c,
                ill_conditioned: v1_ill,
            });
            min_cos = min_cos.min(self.generator.dot(v1).clamp(-1.0, 1.0));
            self.vertices.push(CellVertex {
                pos: v2,
                plane_a: c,
                plane_b: a,
                ill_conditioned: v2_ill,
            });
            min_cos = min_cos.min(self.generator.dot(v2).clamp(-1.0, 1.0));
            self.edge_planes.push(b);
            self.edge_planes.push(c);
            self.edge_planes.push(a);
        } else {
            self.vertices.push(CellVertex {
                pos: v0,
                plane_a: a,
                plane_b: b,
                ill_conditioned: v0_ill,
            });
            min_cos = min_cos.min(self.generator.dot(v0).clamp(-1.0, 1.0));
            self.vertices.push(CellVertex {
                pos: v2,
                plane_a: c,
                plane_b: a,
                ill_conditioned: v2_ill,
            });
            min_cos = min_cos.min(self.generator.dot(v2).clamp(-1.0, 1.0));
            self.vertices.push(CellVertex {
                pos: v1,
                plane_a: b,
                plane_b: c,
                ill_conditioned: v1_ill,
            });
            min_cos = min_cos.min(self.generator.dot(v1).clamp(-1.0, 1.0));
            self.edge_planes.push(a);
            self.edge_planes.push(c);
            self.edge_planes.push(b);
        }

        self.cached_min_vertex_cos = min_cos;
        self.cached_min_vertex_cos_valid = true;
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

        let inside: Vec<bool> = self
            .vertices
            .iter()
            .map(|v| new_plane.signed_distance(v.pos) >= -EPS_PLANE_CLIP)
            .collect();

        let inside_count = inside.iter().filter(|&&x| x).count();
        if inside_count == n {
            return;
        }
        if inside_count == 0 {
            self.vertices.clear();
            self.edge_planes.clear();
            self.dead = true;
            self.cached_min_vertex_cos = 1.0;
            self.cached_min_vertex_cos_valid = false;
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
        let (entry_pos, entry_ill) =
            Self::intersect_two_planes(&self.planes[entry_edge_plane], &new_plane, entry_hint);
        let entry_vertex = CellVertex {
            pos: entry_pos,
            plane_a: entry_edge_plane,
            plane_b: new_plane_idx,
            ill_conditioned: entry_ill,
        };

        let exit_edge_plane = self.edge_planes[exit_idx];
        let exit_start = self.vertices[exit_idx].pos;
        let exit_end = self.vertices[(exit_idx + 1) % n].pos;
        let d_exit_start = new_plane.signed_distance(exit_start);
        let d_exit_end = new_plane.signed_distance(exit_end);
        let t_exit = d_exit_start / (d_exit_start - d_exit_end);
        let exit_hint = (exit_start * (1.0 - t_exit) + exit_end * t_exit).normalize();
        let (exit_pos, exit_ill) =
            Self::intersect_two_planes(&self.planes[exit_edge_plane], &new_plane, exit_hint);
        let exit_vertex = CellVertex {
            pos: exit_pos,
            plane_a: exit_edge_plane,
            plane_b: new_plane_idx,
            ill_conditioned: exit_ill,
        };

        self.tmp_vertices.clear();
        self.tmp_edge_planes.clear();
        self.tmp_vertices.reserve(n);
        self.tmp_edge_planes.reserve(n);

        let mut min_cos = 1.0f32;
        self.tmp_vertices.push(entry_vertex);
        self.tmp_edge_planes.push(entry_edge_plane);
        min_cos = min_cos.min(self.generator.dot(entry_vertex.pos).clamp(-1.0, 1.0));

        let mut i = (entry_idx + 1) % n;
        while i != (exit_idx + 1) % n {
            let v = self.vertices[i];
            min_cos = min_cos.min(self.generator.dot(v.pos).clamp(-1.0, 1.0));
            self.tmp_vertices.push(v);
            self.tmp_edge_planes.push(self.edge_planes[i]);
            i = (i + 1) % n;
        }

        self.tmp_vertices.push(exit_vertex);
        self.tmp_edge_planes.push(new_plane_idx);
        min_cos = min_cos.min(self.generator.dot(exit_vertex.pos).clamp(-1.0, 1.0));

        std::mem::swap(&mut self.vertices, &mut self.tmp_vertices);
        std::mem::swap(&mut self.edge_planes, &mut self.tmp_edge_planes);
        self.cached_min_vertex_cos = min_cos;
        self.cached_min_vertex_cos_valid = true;
    }

    /// Add a new plane (neighbor's bisector) and clip the cell. O(n).
    /// Skips neighbors that are too close to the generator (degenerate bisector).
    pub fn clip(&mut self, neighbor_idx: usize, neighbor: Vec3) {
        // Skip neighbors that are too close - their bisector is numerically unstable
        let diff = self.generator - neighbor;
        if diff.length_squared() < MIN_BISECTOR_DISTANCE * MIN_BISECTOR_DISTANCE {
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

    /// Get the minimum cosine (furthest vertex angle) from generator.
    /// Returns 1.0 if no vertices.
    #[inline]
    pub fn min_vertex_cos(&self) -> f32 {
        if self.cached_min_vertex_cos_valid {
            self.cached_min_vertex_cos
        } else {
            self.vertices
                .iter()
                .map(|v| self.generator.dot(v.pos).clamp(-1.0, 1.0))
                .fold(1.0f32, f32::min)
        }
    }

    /// Check if we can terminate early based on security radius.
    pub fn can_terminate(&self, next_neighbor_cos: f32) -> bool {
        if self.vertices.len() < 3 {
            return false;
        }

        let min_cos = self.min_vertex_cos();

        if min_cos <= 0.0 {
            return false;
        }

        // Conservative bound for epsilon-aware certification:
        // if a vertex is at angle theta from the generator, any generator within
        // (theta + 2*eps) of that vertex must be within (2*theta + 2*eps) of the generator.
        // Use eps_cell as a worst-case bound over vertices to ensure candidate completeness.
        let sin_theta = (1.0 - min_cos * min_cos).max(0.0).sqrt();
        let cos_theta_eps = min_cos * self.cos_eps - sin_theta * self.sin_eps;
        let cos_2max = 2.0 * cos_theta_eps * cos_theta_eps - 1.0;
        next_neighbor_cos < cos_2max - self.termination_margin
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
                    self.cached_min_vertex_cos = 1.0;
                    self.cached_min_vertex_cos_valid = false;

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
        self.cached_min_vertex_cos = 1.0;
        self.cached_min_vertex_cos_valid = false;
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

// =============================================================================
// F64 Cell Builder - High-precision fallback for uncertain vertices
// =============================================================================

/// Epsilon for f64 plane clipping (much tighter than f32).
const F64_EPS_CLIP: f64 = 1e-14;
/// Epsilon for f64 plane parallelism check.
const F64_EPS_PARALLEL: f64 = 1e-12;

/// A great circle in f64 precision.
#[derive(Debug, Clone, Copy)]
struct F64GreatCircle {
    normal: DVec3,
}

impl F64GreatCircle {
    #[inline]
    fn bisector(a: DVec3, b: DVec3) -> Self {
        let normal = (a - b).normalize();
        F64GreatCircle { normal }
    }

    #[inline]
    fn signed_distance(&self, point: DVec3) -> f64 {
        self.normal.dot(point)
    }
}

/// A vertex in the f64 cell polygon.
#[derive(Debug, Clone, Copy)]
struct F64CellVertex {
    pos: DVec3,
    plane_a: usize,
    plane_b: usize,
}

/// High-precision cell builder using f64 arithmetic throughout.
/// Used as fallback when f32 certification fails.
#[derive(Debug, Clone)]
pub struct F64CellBuilder {
    generator_idx: usize,
    generator: DVec3,
    planes: Vec<F64GreatCircle>,
    neighbor_indices: Vec<usize>,
    vertices: Vec<F64CellVertex>,
    edge_planes: Vec<usize>,
    seeded: bool,
    dead: bool,
    cached_min_vertex_cos: f32,
    cached_min_vertex_cos_valid: bool,
    sin_eps: f32,
    cos_eps: f32,
    termination_margin: f32,
}

impl F64CellBuilder {
    /// Create a new f64 cell builder for the given generator.
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        let gen64 = DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64);
        let gen64 = if gen64.length_squared() > 0.0 {
            gen64.normalize()
        } else {
            gen64
        };
        let eps_cell = SUPPORT_VERTEX_ANGLE_EPS + SUPPORT_CLUSTER_RADIUS_ANGLE;
        let (sin_eps, cos_eps) = (eps_cell as f32).sin_cos();
        let termination_margin = EPS_TERMINATION_MARGIN
            + SUPPORT_CERT_MARGIN_ABS as f32
            + 2.0 * SUPPORT_EPS_ABS as f32
            + 2.0 * support_cluster_drift_dot() as f32;
        Self {
            generator_idx,
            generator: gen64,
            planes: Vec::with_capacity(32),
            neighbor_indices: Vec::with_capacity(32),
            vertices: Vec::with_capacity(32),
            edge_planes: Vec::with_capacity(32),
            seeded: false,
            dead: false,
            cached_min_vertex_cos: 1.0,
            cached_min_vertex_cos_valid: false,
            sin_eps,
            cos_eps,
            termination_margin,
        }
    }

    /// Intersect two great circles to find vertex position.
    fn intersect_two_planes(p1: &F64GreatCircle, p2: &F64GreatCircle, hint: DVec3) -> DVec3 {
        let cross = p1.normal.cross(p2.normal);
        let len = cross.length();
        if len < F64_EPS_PARALLEL {
            // Nearly parallel - use hint
            return hint.normalize();
        }
        let dir = cross / len;
        // Choose sign based on hint
        if dir.dot(hint) >= 0.0 {
            dir
        } else {
            -dir
        }
    }

    /// Try to seed from a specific triplet of planes.
    fn try_seed_from_triplet(&mut self, a: usize, b: usize, c: usize) -> bool {
        let v0 = Self::intersect_two_planes(&self.planes[a], &self.planes[b], self.generator);
        let v1 = Self::intersect_two_planes(&self.planes[b], &self.planes[c], self.generator);
        let v2 = Self::intersect_two_planes(&self.planes[c], &self.planes[a], self.generator);

        // Check that all 3 vertices satisfy ALL accumulated half-spaces
        for plane_idx in 0..self.planes.len() {
            if plane_idx == a || plane_idx == b || plane_idx == c {
                continue;
            }
            let plane = &self.planes[plane_idx];
            let d0 = plane.signed_distance(v0);
            let d1 = plane.signed_distance(v1);
            let d2 = plane.signed_distance(v2);
            if d0 < -F64_EPS_CLIP && d1 < -F64_EPS_CLIP && d2 < -F64_EPS_CLIP {
                return false;
            }
        }

        // Verify generator is inside the spherical triangle
        let edge_plane_01 = v0.cross(v1);
        let edge_plane_12 = v1.cross(v2);
        let edge_plane_20 = v2.cross(v0);

        let inside_01 = edge_plane_01.dot(self.generator);
        let inside_12 = edge_plane_12.dot(self.generator);
        let inside_20 = edge_plane_20.dot(self.generator);

        let consistent_winding = (inside_01 > 0.0 && inside_12 > 0.0 && inside_20 > 0.0)
            || (inside_01 < 0.0 && inside_12 < 0.0 && inside_20 < 0.0);

        if !consistent_winding {
            return false;
        }

        self.vertices.clear();
        self.edge_planes.clear();

        // Check winding order - triangle normal should point towards generator
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2);

        let mut min_cos = 1.0f32;
        if normal.dot(self.generator) >= 0.0 {
            // CCW winding when viewed from generator
            self.vertices.push(F64CellVertex {
                pos: v0,
                plane_a: a,
                plane_b: b,
            });
            min_cos = min_cos.min(self.generator.dot(v0).clamp(-1.0, 1.0) as f32);
            self.vertices.push(F64CellVertex {
                pos: v1,
                plane_a: b,
                plane_b: c,
            });
            min_cos = min_cos.min(self.generator.dot(v1).clamp(-1.0, 1.0) as f32);
            self.vertices.push(F64CellVertex {
                pos: v2,
                plane_a: c,
                plane_b: a,
            });
            min_cos = min_cos.min(self.generator.dot(v2).clamp(-1.0, 1.0) as f32);
            self.edge_planes.push(b);
            self.edge_planes.push(c);
            self.edge_planes.push(a);
        } else {
            // Reverse winding
            self.vertices.push(F64CellVertex {
                pos: v0,
                plane_a: a,
                plane_b: b,
            });
            min_cos = min_cos.min(self.generator.dot(v0).clamp(-1.0, 1.0) as f32);
            self.vertices.push(F64CellVertex {
                pos: v2,
                plane_a: c,
                plane_b: a,
            });
            min_cos = min_cos.min(self.generator.dot(v2).clamp(-1.0, 1.0) as f32);
            self.vertices.push(F64CellVertex {
                pos: v1,
                plane_a: b,
                plane_b: c,
            });
            min_cos = min_cos.min(self.generator.dot(v1).clamp(-1.0, 1.0) as f32);
            self.edge_planes.push(a);
            self.edge_planes.push(c);
            self.edge_planes.push(b);
        }

        self.cached_min_vertex_cos = min_cos;
        self.cached_min_vertex_cos_valid = true;
        true
    }

    /// Try to seed the polygon by finding a valid triplet among accumulated planes.
    fn try_seed(&mut self) -> bool {
        let n = self.planes.len();
        if n < 3 {
            return false;
        }

        // Try triplets including the newest plane first
        for i in 0..(n - 1) {
            for j in (i + 1)..(n - 1) {
                if self.try_seed_from_triplet(i, j, n - 1) {
                    return true;
                }
            }
        }

        // Try all other triplets
        for a in 0..(n - 1) {
            for b in (a + 1)..(n - 1) {
                for c in (b + 1)..(n - 1) {
                    if self.try_seed_from_triplet(a, b, c) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Clip the cell by a new plane.
    fn clip_with_plane(&mut self, plane_idx: usize) {
        let new_plane = self.planes[plane_idx];
        let n = self.vertices.len();
        if n < 3 {
            return;
        }

        let inside: Vec<bool> = self
            .vertices
            .iter()
            .map(|v| new_plane.signed_distance(v.pos) >= -F64_EPS_CLIP)
            .collect();

        let inside_count = inside.iter().filter(|&&x| x).count();
        if inside_count == n {
            return; // All inside, no clipping needed
        }
        if inside_count == 0 {
            self.vertices.clear();
            self.edge_planes.clear();
            self.dead = true;
            self.cached_min_vertex_cos = 1.0;
            self.cached_min_vertex_cos_valid = false;
            return;
        }

        // Find entry and exit points
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

        // Compute entry vertex
        let entry_edge_plane = self.edge_planes[entry_idx];
        let entry_start = self.vertices[entry_idx].pos;
        let entry_end = self.vertices[(entry_idx + 1) % n].pos;
        let d_start = new_plane.signed_distance(entry_start);
        let d_end = new_plane.signed_distance(entry_end);
        let t = d_start / (d_start - d_end);
        let entry_hint = (entry_start * (1.0 - t) + entry_end * t).normalize();
        let entry_pos =
            Self::intersect_two_planes(&self.planes[entry_edge_plane], &new_plane, entry_hint);
        let entry_vertex = F64CellVertex {
            pos: entry_pos,
            plane_a: entry_edge_plane,
            plane_b: plane_idx,
        };

        // Compute exit vertex
        let exit_edge_plane = self.edge_planes[exit_idx];
        let exit_start = self.vertices[exit_idx].pos;
        let exit_end = self.vertices[(exit_idx + 1) % n].pos;
        let d_start = new_plane.signed_distance(exit_start);
        let d_end = new_plane.signed_distance(exit_end);
        let t = d_start / (d_start - d_end);
        let exit_hint = (exit_start * (1.0 - t) + exit_end * t).normalize();
        let exit_pos =
            Self::intersect_two_planes(&self.planes[exit_edge_plane], &new_plane, exit_hint);
        let exit_vertex = F64CellVertex {
            pos: exit_pos,
            plane_a: exit_edge_plane,
            plane_b: plane_idx,
        };

        // Build new vertex list
        let mut new_vertices = Vec::with_capacity(n);
        let mut new_edge_planes = Vec::with_capacity(n);

        new_vertices.push(entry_vertex);
        new_edge_planes.push(entry_edge_plane);
        let mut min_cos = self
            .generator
            .dot(entry_vertex.pos)
            .clamp(-1.0, 1.0) as f32;

        let mut i = (entry_idx + 1) % n;
        while i != (exit_idx + 1) % n {
            let v = self.vertices[i];
            min_cos = min_cos.min(self.generator.dot(v.pos).clamp(-1.0, 1.0) as f32);
            new_vertices.push(v);
            new_edge_planes.push(self.edge_planes[i]);
            i = (i + 1) % n;
        }

        new_vertices.push(exit_vertex);
        new_edge_planes.push(plane_idx);
        min_cos = min_cos.min(self.generator.dot(exit_vertex.pos).clamp(-1.0, 1.0) as f32);

        self.vertices = new_vertices;
        self.edge_planes = new_edge_planes;
        self.cached_min_vertex_cos = min_cos;
        self.cached_min_vertex_cos_valid = true;
    }

    /// Add a neighbor and clip the cell.
    pub fn clip(&mut self, neighbor_idx: usize, neighbor: Vec3) {
        if self.dead {
            return;
        }

        let n64 = DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64);
        let n64 = if n64.length_squared() > 0.0 {
            n64.normalize()
        } else {
            n64
        };

        // Skip degenerate bisectors
        let diff = self.generator - n64;
        if diff.length_squared() < (MIN_BISECTOR_DISTANCE as f64).powi(2) {
            return;
        }

        let new_plane = F64GreatCircle::bisector(self.generator, n64);
        let plane_idx = self.planes.len();
        self.planes.push(new_plane);
        self.neighbor_indices.push(neighbor_idx);

        if !self.seeded {
            if self.planes.len() >= 3 {
                if self.try_seed() {
                    self.seeded = true;
                    // Clip with all non-seed planes
                    for idx in 0..self.planes.len() {
                        // Skip planes that are part of current vertices
                        let is_seed_plane = self
                            .vertices
                            .iter()
                            .any(|v| v.plane_a == idx || v.plane_b == idx);
                        if !is_seed_plane {
                            self.clip_with_plane(idx);
                            if self.dead {
                                return;
                            }
                        }
                    }
                }
            }
            return;
        }

        self.clip_with_plane(plane_idx);
    }

    /// Check if the cell is dead (empty intersection).
    pub fn is_dead(&self) -> bool {
        self.dead
    }

    /// Get vertex count.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get plane count (number of neighbors clipped).
    #[inline]
    pub fn planes_count(&self) -> usize {
        self.planes.len()
    }

    /// Returns true if this neighbor has already been clipped into the cell.
    #[inline]
    pub fn has_neighbor(&self, neighbor_idx: usize) -> bool {
        self.neighbor_indices.contains(&neighbor_idx)
    }

    /// Returns an iterator over the neighbor indices that have been clipped.
    #[inline]
    pub fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbor_indices.iter().copied()
    }

    /// Get the minimum cosine (furthest vertex angle) from generator.
    /// Returns 1.0 if no vertices.
    #[inline]
    pub fn min_vertex_cos(&self) -> f64 {
        if self.cached_min_vertex_cos_valid {
            self.cached_min_vertex_cos as f64
        } else {
            self.vertices
                .iter()
                .map(|v| self.generator.dot(v.pos).clamp(-1.0, 1.0))
                .fold(1.0f64, f64::min)
        }
    }

    /// Check if we can terminate early based on security radius.
    pub fn can_terminate(&self, next_neighbor_cos: f32) -> bool {
        if self.vertices.len() < 3 {
            return false;
        }

        let min_cos = self.min_vertex_cos() as f32;

        if min_cos <= 0.0 {
            return false;
        }

        // Conservative bound for epsilon-aware certification:
        // if a vertex is at angle theta from the generator, any generator within
        // (theta + 2*eps) of that vertex must be within (2*theta + 2*eps) of the generator.
        // Use eps_cell as a worst-case bound over vertices to ensure candidate completeness.
        let sin_theta = (1.0 - min_cos * min_cos).max(0.0).sqrt();
        let cos_theta_eps = min_cos * self.cos_eps - sin_theta * self.sin_eps;
        let cos_2max = 2.0 * cos_theta_eps * cos_theta_eps - 1.0;
        next_neighbor_cos < cos_2max - self.termination_margin
    }

    /// Attempt to reseed using the best-conditioned triplet among all planes.
    /// Returns true if a non-degenerate cell is recovered.
    pub fn try_reseed_best(&mut self) -> bool {
        let plane_count = self.planes.len();
        if plane_count < 3 {
            return false;
        }

        let min_score = (F64_EPS_PARALLEL * F64_EPS_PARALLEL) as f64;
        let mut best_triplet: Option<(usize, usize, usize)> = None;
        let mut best_score = 0.0f64;

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
                    self.cached_min_vertex_cos = 1.0;
                    self.cached_min_vertex_cos_valid = false;

                    if !self.try_seed_from_triplet(a, b, c) {
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
        self.cached_min_vertex_cos = 1.0;
        self.cached_min_vertex_cos_valid = false;
        if !self.try_seed_from_triplet(a, b, c) {
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

    /// Reset the builder for a new cell.
    pub fn reset(&mut self, generator_idx: usize, generator: Vec3) {
        let gen64 = DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64);
        let gen64 = if gen64.length_squared() > 0.0 {
            gen64.normalize()
        } else {
            gen64
        };
        self.generator_idx = generator_idx;
        self.generator = gen64;
        self.planes.clear();
        self.neighbor_indices.clear();
        self.vertices.clear();
        self.edge_planes.clear();
        self.seeded = false;
        self.dead = false;
        self.cached_min_vertex_cos = 1.0;
        self.cached_min_vertex_cos_valid = false;
    }

    /// Iterator over vertices for testing/inspection.
    /// Returns (vertex_index, position, plane_a, plane_b) for each vertex.
    pub fn vertices_iter(&self) -> impl Iterator<Item = (usize, DVec3, usize, usize)> + '_ {
        self.vertices
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.pos, v.plane_a, v.plane_b))
    }

    /// Get the plane normal for a given plane index.
    pub fn plane_normal(&self, plane_idx: usize) -> DVec3 {
        self.planes[plane_idx].normal
    }

    /// Get the neighbor index for a given plane index.
    pub fn neighbor_index(&self, plane_idx: usize) -> usize {
        self.neighbor_indices[plane_idx]
    }

    /// Compute the gap (slack) from each vertex to all non-defining generators.
    ///
    /// For each vertex V defined by planes (A, B), the gap to generator C is:
    ///   gap_C = dot(V, G) - dot(V, C)
    ///
    /// where G is the cell's generator. A positive gap means C is farther than G
    /// (C is outside the support set). The minimum positive gap tells us how close
    /// the nearest excluded generator is.
    ///
    /// Returns for each vertex: (min_gap, support_set)
    /// - min_gap: smallest positive gap (closest excluded generator)
    /// - support_set: generators with gap <= eps_support (including defining generators)
    pub fn compute_vertex_gaps(&self, points: &[Vec3], eps_support: f64) -> Vec<(f64, Vec<u32>)> {
        if self.dead || self.vertices.len() < 3 {
            return Vec::new();
        }

        let g = self.generator;
        let gen_idx = self.generator_idx as u32;

        // Pre-normalize all neighbor positions once (instead of per-vertex)
        let neighbor_positions: Vec<DVec3> = self
            .neighbor_indices
            .iter()
            .map(|&idx| {
                let p = points[idx];
                DVec3::new(p.x as f64, p.y as f64, p.z as f64).normalize()
            })
            .collect();

        self.vertices
            .iter()
            .map(|v| {
                let v_pos = v.pos.normalize();
                let dot_g = v_pos.dot(g);

                // Defining generators (always in support set)
                let def_a = self.neighbor_indices[v.plane_a] as u32;
                let def_b = self.neighbor_indices[v.plane_b] as u32;

                let mut min_gap = f64::INFINITY;
                let mut support_set = vec![gen_idx, def_a, def_b];

                // Check all other generators
                for (plane_idx, c) in neighbor_positions.iter().enumerate() {
                    // Skip defining planes
                    if plane_idx == v.plane_a || plane_idx == v.plane_b {
                        continue;
                    }

                    let dot_c = v_pos.dot(*c);
                    let gap = dot_g - dot_c;

                    if gap <= eps_support {
                        // This generator is in the support set (near-degenerate)
                        support_set.push(self.neighbor_indices[plane_idx] as u32);
                    } else if gap < min_gap {
                        // Track closest excluded generator
                        min_gap = gap;
                    }
                }

                support_set.sort_unstable();
                support_set.dedup();

                (min_gap, support_set)
            })
            .collect()
    }

    /// Compute the conditioning (sin of angle between defining planes) for a vertex.
    fn vertex_conditioning(&self, v: &F64CellVertex) -> f64 {
        let n_a = self.planes[v.plane_a].normal;
        let n_b = self.planes[v.plane_b].normal;
        n_a.cross(n_b).length()
    }

    /// Compute the conditioning for a vertex by index (for testing).
    pub fn vertex_conditioning_by_index(&self, idx: usize) -> f64 {
        self.vertex_conditioning(&self.vertices[idx])
    }

    /// Convert f64 vertices to f32 VertexData with certified support keys.
    ///
    /// Uses slack-based certification: for each vertex, we compute the gap to all
    /// non-defining generators. The support set includes generators with gap <= eps_support.
    /// Certification passes if min_gap > eps_support + error_margin.
    ///
    /// Panics if certification fails, since f64 should be able to certify any vertex.
    pub fn to_vertex_data(&self, points: &[Vec3], support_data: &mut Vec<u32>) -> Vec<VertexData> {
        if self.dead || self.vertices.len() < 3 {
            return Vec::new();
        }

        // Compute gaps and support sets for all vertices
        let vertex_gaps = self.compute_vertex_gaps(points, SUPPORT_EPS_ABS);

        // F64 vertex error bound (Case A from plan2.md):
        // vertex_err ≈ C * f64::EPSILON / conditioning
        // where conditioning = sin(angle between defining planes)
        //
        // Gap error = vertex_err * |G - C| (generator distance)
        // For certification: min_gap > SUPPORT_EPS_ABS + gap_error_bound
        //
        // We use C = 16 to account for multiple f64 operations (normalize, cross, etc.)
        const F64_VERTEX_ERR_FACTOR: f64 = 16.0 * f64::EPSILON;

        self.vertices
            .iter()
            .zip(vertex_gaps.iter())
            .enumerate()
            .map(|(vertex_idx, (v, (min_gap, support_set)))| {
                let vertex_dir = v.pos.normalize();
                let pos = Vec3::new(
                    vertex_dir.x as f32,
                    vertex_dir.y as f32,
                    vertex_dir.z as f32,
                );

                // Per-vertex conditioning-based error bound
                let conditioning = self.vertex_conditioning(v).max(1e-6);
                let vertex_err = F64_VERTEX_ERR_FACTOR / conditioning;
                // Gap error is vertex_err * max_generator_distance
                // Conservative: use 2.0 as upper bound for |G - C| on unit sphere
                let gap_err_bound = vertex_err * 2.0;
                let cert_threshold = SUPPORT_EPS_ABS + gap_err_bound;

                // Certification: min_gap must be larger than threshold
                // This ensures no excluded generator could enter the support set under uncertainty
                let certified = *min_gap > cert_threshold;

                if !certified {
                    let plane_sin = self.vertex_conditioning(v);
                    panic!(
                        "f64 slack certification failed for cell {} vertex {} \
                        (min_gap={:.2e}, threshold={:.2e}, plane_sin={:.2e}, support={:?})",
                        self.generator_idx,
                        vertex_idx,
                        min_gap,
                        cert_threshold,
                        plane_sin,
                        support_set
                    );
                }

                let key = if support_set.len() == 3 {
                    VertexKey::Triplet([support_set[0], support_set[1], support_set[2]])
                } else if support_set.len() >= 4 {
                    let start = support_data.len() as u32;
                    support_data.extend(support_set.iter().copied());
                    let len = support_set.len() as u8;
                    VertexKey::Support { start, len }
                } else {
                    panic!(
                        "f64 produced support set with {} elements for cell {} vertex {}",
                        support_set.len(),
                        self.generator_idx,
                        vertex_idx
                    );
                };

                (key, pos)
            })
            .collect()
    }
}
