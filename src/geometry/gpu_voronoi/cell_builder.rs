//! Cell building types and algorithms for spherical Voronoi computation.

use glam::{DVec3, Vec3};

use super::constants::{
    EPS_PLANE_CLIP,
    EPS_PLANE_CONDITION,
    EPS_PLANE_CONTAINS,
    EPS_PLANE_PARALLEL,
    EPS_TERMINATION_MARGIN,
    MIN_BISECTOR_DISTANCE,
    SUPPORT_CERT_MARGIN_ABS,
    SUPPORT_EPS_ABS,
    SUPPORT_CLUSTER_RADIUS_ANGLE,
    SUPPORT_VERTEX_ANGLE_EPS,
    support_cluster_drift_dot,
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
        let seed = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
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
}

impl IncrementalCellBuilder {
    #[inline]
    fn canonical_vertex_dir(
        &self,
        points: &[Vec3],
        v: &CellVertex,
    ) -> (DVec3, f64, bool, f64) {
        #[inline]
        fn angle_between_unit(a: DVec3, b: DVec3) -> f64 {
            let dot = a.dot(b).clamp(-1.0, 1.0);
            dot.acos()
        }

        let gen = points[self.generator_idx];
        let na_idx = self.neighbor_indices[v.plane_a];
        let nb_idx = self.neighbor_indices[v.plane_b];
        let na = points[na_idx];
        let nb = points[nb_idx];

        // Treat the f32 positions as canonical (no renormalization) to stay
        // consistent with the f32 plane normals used during clipping.
        let g64 = DVec3::new(gen.x as f64, gen.y as f64, gen.z as f64);
        let na64 = DVec3::new(na.x as f64, na.y as f64, na.z as f64);
        let nb64 = DVec3::new(nb.x as f64, nb.y as f64, nb.z as f64);

        let n_a = (g64 - na64).normalize();
        let n_b = (g64 - nb64).normalize();
        let cross = n_a.cross(n_b);
        let len = cross.length();

        let mut ill_conditioned = v.ill_conditioned;
        if len < EPS_PLANE_PARALLEL as f64 {
            ill_conditioned = true;
        }

        let mut v64 = if len > 0.0 {
            cross / len
        } else {
            DVec3::new(v.pos.x as f64, v.pos.y as f64, v.pos.z as f64).normalize()
        };

        let ref_sum = g64 + na64 + nb64;
        let ref_dir = if ref_sum.length_squared() > 0.0 {
            ref_sum.normalize()
        } else {
            g64
        };
        if v64.dot(ref_dir) < 0.0 {
            v64 = -v64;
        }

        let v32 = {
            let v32 = DVec3::new(v.pos.x as f64, v.pos.y as f64, v.pos.z as f64);
            if v32.length_squared() > 0.0 {
                v32.normalize()
            } else {
                v64
            }
        };
        let angle = angle_between_unit(v64, v32);
        let drift_angle = if angle.is_finite() {
            if angle > SUPPORT_CLUSTER_RADIUS_ANGLE {
                ill_conditioned = true;
            }
            angle.min(SUPPORT_CLUSTER_RADIUS_ANGLE)
        } else {
            ill_conditioned = true;
            SUPPORT_CLUSTER_RADIUS_ANGLE
        };

        // Formal plane-error bound: error in plane normals scaled by conditioning.
        let na_f32 = self.planes[v.plane_a].normal;
        let nb_f32 = self.planes[v.plane_b].normal;
        let na_f32 = DVec3::new(na_f32.x as f64, na_f32.y as f64, na_f32.z as f64).normalize();
        let nb_f32 = DVec3::new(nb_f32.x as f64, nb_f32.y as f64, nb_f32.z as f64).normalize();
        let plane_err = angle_between_unit(na_f32, n_a) + angle_between_unit(nb_f32, n_b);
        let plane_sin = len.max(EPS_PLANE_PARALLEL as f64);
        let r_est = plane_err / plane_sin;

        (v64, drift_angle, ill_conditioned, r_est)
    }
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

        if normal.dot(self.generator) >= 0.0 {
            self.vertices.push(CellVertex { pos: v0, plane_a: a, plane_b: b, ill_conditioned: v0_ill });
            self.vertices.push(CellVertex { pos: v1, plane_a: b, plane_b: c, ill_conditioned: v1_ill });
            self.vertices.push(CellVertex { pos: v2, plane_a: c, plane_b: a, ill_conditioned: v2_ill });
            self.edge_planes.push(b);
            self.edge_planes.push(c);
            self.edge_planes.push(a);
        } else {
            self.vertices.push(CellVertex { pos: v0, plane_a: a, plane_b: b, ill_conditioned: v0_ill });
            self.vertices.push(CellVertex { pos: v2, plane_a: c, plane_b: a, ill_conditioned: v2_ill });
            self.vertices.push(CellVertex { pos: v1, plane_a: b, plane_b: c, ill_conditioned: v1_ill });
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
        let (entry_pos, entry_ill) = Self::intersect_two_planes(
            &self.planes[entry_edge_plane],
            &new_plane,
            entry_hint,
        );
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
        let (exit_pos, exit_ill) = Self::intersect_two_planes(
            &self.planes[exit_edge_plane],
            &new_plane,
            exit_hint,
        );
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

        // Conservative bound for epsilon-aware certification:
        // if a vertex is at angle theta from the generator, any generator within
        // (theta + 2*eps) of that vertex must be within (2*theta + 2*eps) of the generator.
        // Use eps_cell as a worst-case bound over vertices to ensure candidate completeness.
        let eps_cell = SUPPORT_VERTEX_ANGLE_EPS + SUPPORT_CLUSTER_RADIUS_ANGLE;
        let sin_theta = (1.0 - min_cos * min_cos).max(0.0).sqrt();
        let (sin_eps, cos_eps) = (eps_cell as f32).sin_cos();
        let cos_theta_eps = min_cos * cos_eps - sin_theta * sin_eps;
        let cos_2max = 2.0 * cos_theta_eps * cos_theta_eps - 1.0;
        // Conservatively include support/certification margins in dot space.
        let termination_margin = EPS_TERMINATION_MARGIN
            + SUPPORT_CERT_MARGIN_ABS as f32
            + 2.0 * SUPPORT_EPS_ABS as f32
            + 2.0 * support_cluster_drift_dot() as f32;
        next_neighbor_cos < cos_2max - termination_margin
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

    /// Write vertices with canonical keys into the provided buffer.
    /// Returns the number of vertices written.
    ///
    /// Failed vertex indices are recorded with reason codes:
    /// - 1 = ill-conditioned only
    /// - 2 = gap only
    /// - 3 = both ill-conditioned and gap
    #[inline]
    pub fn get_vertices_into(
        &self,
        points: &[Vec3],
        support_eps: f64,
        candidates_complete: bool,
        support_data: &mut Vec<u32>,
        out: &mut Vec<VertexData>,
        cert_checked: &mut bool,
        cert_failed: &mut bool,
        cert_checked_vertices: &mut usize,
        cert_failed_vertices: &mut usize,
        cert_failed_ill_vertices: &mut usize,
        cert_failed_gap_vertices: &mut usize,
        gap_sampler: &mut GapSampler,
        cert_failed_vertex_indices: &mut Vec<(u32, u8)>,
    ) -> usize {
        #[inline]
        fn sort3(mut a: [u32; 3]) -> [u32; 3] {
            if a[0] > a[1] { a.swap(0, 1); }
            if a[1] > a[2] { a.swap(1, 2); }
            if a[0] > a[1] { a.swap(0, 1); }
            a
        }

        out.reserve(self.vertices.len());
        let count = self.vertices.len();

        let mut candidates: Vec<u32> = Vec::new();
        let mut candidate_positions: Vec<DVec3> = Vec::new();
        if candidates_complete && support_eps > 0.0 {
            candidates.reserve(self.neighbor_indices.len() + 1);
            candidates.push(self.generator_idx as u32);
            candidates.extend(self.neighbor_indices.iter().map(|&idx| idx as u32));
            candidate_positions = candidates
                .iter()
                .map(|&idx| {
                    let p = points[idx as usize];
                    DVec3::new(p.x as f64, p.y as f64, p.z as f64)
                })
                .collect();
        }

        let mut support: Vec<u32> = Vec::with_capacity(self.planes.len() + 1);
        let do_support = candidates_complete && support_eps > 0.0;
        if do_support && !self.vertices.is_empty() {
            *cert_checked = true;
        }
        for v in &self.vertices {
            // Track vertex index before pushing
            let vertex_idx = out.len() as u32;
            let key = if do_support {
                support.clear();
                let mut fail_ill = false;
                let mut fail_gap = false;
                let (v64, drift_angle, ill_conditioned, plane_err) =
                    self.canonical_vertex_dir(points, v);
                // Per-vertex epsilon (angle) derived from FP error + observed drift.
                let eps_angle: f64 = SUPPORT_VERTEX_ANGLE_EPS + drift_angle + plane_err;
                let eps_cell = SUPPORT_VERTEX_ANGLE_EPS + SUPPORT_CLUSTER_RADIUS_ANGLE;
                let cert_threshold = SUPPORT_CERT_MARGIN_ABS;
                if ill_conditioned {
                    *cert_failed = true;
                    fail_ill = true;
                    *cert_failed_ill_vertices += 1;
                }
                if eps_angle > eps_cell {
                    *cert_failed = true;
                    if !fail_ill {
                        *cert_failed_gap_vertices += 1;
                    }
                    fail_gap = true;
                }
                let mut max_dot = f64::NEG_INFINITY;
                for g in &candidate_positions {
                    let d = v64.dot(*g);
                    if d > max_dot {
                        max_dot = d;
                    }
                }

                let max_dot = max_dot.clamp(-1.0, 1.0);
                let sin_theta = (1.0 - max_dot * max_dot).max(0.0).sqrt();
                let angle2: f64 = 2.0 * eps_angle;
                let angle3: f64 = 3.0 * eps_angle;
                let (sin2, cos2) = angle2.sin_cos();
                let (sin3, cos3) = angle3.sin_cos();
                let cluster_dot = max_dot * cos2 - sin_theta * sin2;
                let gap_dot = max_dot * cos3 - sin_theta * sin3;
                let cluster_delta = (max_dot - cluster_dot).max(0.0f64) + support_eps;
                let mut gap_delta = (max_dot - gap_dot).max(0.0f64) + support_eps;
                if gap_delta < cluster_delta {
                    gap_delta = cluster_delta;
                }

                let mut best_outside = f64::NEG_INFINITY;
                for (idx, g) in candidate_positions.iter().enumerate() {
                    let d = v64.dot(*g);
                    if max_dot - d <= cluster_delta {
                        support.push(candidates[idx]);
                    } else if d > best_outside {
                        best_outside = d;
                    }
                }

                let max_dot_min = max_dot - cluster_delta;
                if !fail_gap && best_outside.is_finite() && max_dot - best_outside <= gap_delta + cert_threshold {
                    *cert_failed = true;
                    if !fail_ill {
                        *cert_failed_gap_vertices += 1;
                    }
                    fail_gap = true;
                }
                if best_outside.is_finite() {
                    gap_sampler.record(max_dot_min - best_outside);
                }
                *cert_checked_vertices += 1;
                let vertex_failed = fail_ill || fail_gap;
                if vertex_failed {
                    *cert_failed_vertices += 1;
                    // Reason code: 1=ill_cond, 2=gap, 3=both
                    let reason = (fail_ill as u8) | ((fail_gap as u8) << 1);
                    cert_failed_vertex_indices.push((vertex_idx, reason));
                }

                let gen = self.generator_idx as u32;
                let na = self.neighbor_indices[v.plane_a] as u32;
                let nb = self.neighbor_indices[v.plane_b] as u32;
                support.push(gen);
                support.push(na);
                support.push(nb);
                support.sort_unstable();
                support.dedup();

                if support.len() >= 4 {
                    debug_assert!(support.len() <= u8::MAX as usize, "support list too large");
                    let start = support_data.len() as u32;
                    support_data.extend(support.iter().copied());
                    let len = support.len() as u8;
                    VertexKey::Support { start, len }
                } else if support.len() == 3 {
                    VertexKey::Triplet([support[0], support[1], support[2]])
                } else {
                    let triplet = sort3([gen, na, nb]);
                    VertexKey::Triplet(triplet)
                }
            } else {
                let triplet = sort3([
                    self.generator_idx as u32,
                    self.neighbor_indices[v.plane_a] as u32,
                    self.neighbor_indices[v.plane_b] as u32,
                ]);
                VertexKey::Triplet(triplet)
            };
            out.push((key, v.pos));
        }
        count
    }

    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    #[inline]
    pub fn min_vertex_generator_dot(&self) -> Option<f32> {
        if self.vertices.len() < 3 {
            return None;
        }
        Some(self.vertices.iter()
            .map(|v| self.generator.dot(v.pos).clamp(-1.0, 1.0))
            .fold(1.0f32, f32::min))
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
