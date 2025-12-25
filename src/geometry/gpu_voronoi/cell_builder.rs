//! Cell building types and algorithms for spherical Voronoi computation.

use glam::{DVec3, Vec3};

use super::constants::{
    support_cluster_drift_dot, EPS_TERMINATION_MARGIN, MIN_BISECTOR_DISTANCE,
    SUPPORT_CERT_MARGIN_ABS, SUPPORT_CLUSTER_RADIUS_ANGLE, SUPPORT_EPS_ABS, SUPPORT_VERTEX_ANGLE_EPS,
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

/// Maximum number of planes (great circle boundaries) per cell.
pub const MAX_PLANES: usize = 32;

/// Maximum number of vertices (plane triplet intersections) per cell.
pub const MAX_VERTICES: usize = 16;

/// Reasons a cell build can fail, requiring fallback to a different algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellFailure {
    /// Exceeded MAX_PLANES during clipping.
    TooManyPlanes,
    /// Exceeded MAX_VERTICES during clipping.
    TooManyVertices,
    /// Cell was completely clipped away (all vertices outside a plane).
    ClippedAway,
    /// Failed to find valid seed triplet.
    NoValidSeed,
    /// Vertex certification failed (support set ambiguous).
    CertificationFailed,
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

/// High-precision cell builder using f64 arithmetic throughout.
/// Used as fallback when f32 certification fails.
///
/// Uses fixed-size stack arrays instead of Vec for better cache locality
/// and to avoid heap allocations. Vertices are stored in SoA layout.
#[derive(Debug, Clone)]
pub struct F64CellBuilder {
    generator_idx: usize,
    generator: DVec3,

    // Planes - stored as normals (DVec3 is simpler than SoA for 3 elements)
    plane_normals: [DVec3; MAX_PLANES],
    neighbor_indices: [usize; MAX_PLANES],
    plane_count: usize,

    // Vertices as SoA (Structure of Arrays) for cache-friendly access
    vertex_x: [f64; MAX_VERTICES],
    vertex_y: [f64; MAX_VERTICES],
    vertex_z: [f64; MAX_VERTICES],
    vertex_plane_a: [usize; MAX_VERTICES],
    vertex_plane_b: [usize; MAX_VERTICES],
    vertex_count: usize,

    // Edge planes (parallel to vertices array)
    edge_planes: [usize; MAX_VERTICES],

    // Scratch buffers for clipping operations
    scratch_inside: [bool; MAX_VERTICES],
    scratch_vertex_x: [f64; MAX_VERTICES],
    scratch_vertex_y: [f64; MAX_VERTICES],
    scratch_vertex_z: [f64; MAX_VERTICES],
    scratch_vertex_plane_a: [usize; MAX_VERTICES],
    scratch_vertex_plane_b: [usize; MAX_VERTICES],
    scratch_vertex_count: usize,
    scratch_edge_planes: [usize; MAX_VERTICES],

    seeded: bool,
    failed: Option<CellFailure>,
    cached_min_vertex_cos: f32,
    cached_min_vertex_cos_valid: bool,
    sin_eps: f32,
    cos_eps: f32,
    termination_margin: f32,
}

impl F64CellBuilder {
    #[inline(always)]
    fn sort3(a: &mut u32, b: &mut u32, c: &mut u32) {
        if *a > *b {
            std::mem::swap(a, b);
        }
        if *b > *c {
            std::mem::swap(b, c);
        }
        if *a > *b {
            std::mem::swap(a, b);
        }
    }

    #[inline(always)]
    fn debug_assert_unitish(v: DVec3) {
        #[cfg(debug_assertions)]
        {
            let ls = v.length_squared();
            debug_assert!(
                ls.is_finite() && (ls - 1.0).abs() <= 1e-5,
                "expected ~unit vector; len_sq={}",
                ls
            );
        }
    }

    #[inline(always)]
    fn debug_assert_unit(v: DVec3) {
        debug_assert!(
            (v.length_squared() - 1.0).abs() <= 1e-10,
            "expected unit vector; len_sq={}",
            v.length_squared()
        );
    }

    // =========================================================================
    // Helper methods for SoA vertex access
    // =========================================================================

    /// Get vertex position as DVec3.
    #[inline]
    fn get_vertex_pos(&self, i: usize) -> DVec3 {
        DVec3::new(self.vertex_x[i], self.vertex_y[i], self.vertex_z[i])
    }

    /// Get plane as F64GreatCircle.
    #[inline]
    fn get_plane(&self, i: usize) -> F64GreatCircle {
        F64GreatCircle {
            normal: self.plane_normals[i],
        }
    }

    /// Set vertex at index.
    #[inline]
    fn set_vertex(&mut self, i: usize, pos: DVec3, plane_a: usize, plane_b: usize) {
        self.vertex_x[i] = pos.x;
        self.vertex_y[i] = pos.y;
        self.vertex_z[i] = pos.z;
        self.vertex_plane_a[i] = plane_a;
        self.vertex_plane_b[i] = plane_b;
    }

    /// Push a new vertex, incrementing count.
    #[inline]
    fn push_vertex(&mut self, pos: DVec3, plane_a: usize, plane_b: usize) {
        let i = self.vertex_count;
        self.set_vertex(i, pos, plane_a, plane_b);
        self.vertex_count += 1;
    }

    /// Push vertex to scratch buffer. Returns Err if scratch buffer is full.
    #[inline]
    fn push_scratch_vertex(&mut self, pos: DVec3, plane_a: usize, plane_b: usize) -> Result<(), CellFailure> {
        let i = self.scratch_vertex_count;
        if i >= MAX_VERTICES {
            self.failed = Some(CellFailure::TooManyVertices);
            return Err(CellFailure::TooManyVertices);
        }
        self.scratch_vertex_x[i] = pos.x;
        self.scratch_vertex_y[i] = pos.y;
        self.scratch_vertex_z[i] = pos.z;
        self.scratch_vertex_plane_a[i] = plane_a;
        self.scratch_vertex_plane_b[i] = plane_b;
        self.scratch_vertex_count += 1;
        Ok(())
    }

    /// Get scratch vertex position.
    #[inline]
    fn get_scratch_vertex_pos(&self, i: usize) -> DVec3 {
        DVec3::new(
            self.scratch_vertex_x[i],
            self.scratch_vertex_y[i],
            self.scratch_vertex_z[i],
        )
    }

    /// Copy scratch buffer to vertices (scratch may have more capacity).
    #[inline]
    fn copy_scratch_to_vertices(&mut self) {
        let n = self.scratch_vertex_count;
        self.vertex_x[..n].copy_from_slice(&self.scratch_vertex_x[..n]);
        self.vertex_y[..n].copy_from_slice(&self.scratch_vertex_y[..n]);
        self.vertex_z[..n].copy_from_slice(&self.scratch_vertex_z[..n]);
        self.vertex_plane_a[..n].copy_from_slice(&self.scratch_vertex_plane_a[..n]);
        self.vertex_plane_b[..n].copy_from_slice(&self.scratch_vertex_plane_b[..n]);
        self.vertex_count = n;
        self.edge_planes[..n].copy_from_slice(&self.scratch_edge_planes[..n]);
    }

    /// Create a new f64 cell builder for the given generator.
    pub fn new(generator_idx: usize, generator: Vec3) -> Self {
        let gen64 = DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64).normalize();
        let eps_cell = SUPPORT_VERTEX_ANGLE_EPS + SUPPORT_CLUSTER_RADIUS_ANGLE;
        let (sin_eps, cos_eps) = (eps_cell as f32).sin_cos();
        let termination_margin = EPS_TERMINATION_MARGIN
            + SUPPORT_CERT_MARGIN_ABS as f32
            + 2.0 * SUPPORT_EPS_ABS as f32
            + 2.0 * support_cluster_drift_dot() as f32;
        Self {
            generator_idx,
            generator: gen64,

            // Planes
            plane_normals: [DVec3::ZERO; MAX_PLANES],
            neighbor_indices: [0; MAX_PLANES],
            plane_count: 0,

            // Vertices (SoA)
            vertex_x: [0.0; MAX_VERTICES],
            vertex_y: [0.0; MAX_VERTICES],
            vertex_z: [0.0; MAX_VERTICES],
            vertex_plane_a: [0; MAX_VERTICES],
            vertex_plane_b: [0; MAX_VERTICES],
            vertex_count: 0,

            // Edge planes
            edge_planes: [0; MAX_VERTICES],

            // Scratch buffers
            scratch_inside: [false; MAX_VERTICES],
            scratch_vertex_x: [0.0; MAX_VERTICES],
            scratch_vertex_y: [0.0; MAX_VERTICES],
            scratch_vertex_z: [0.0; MAX_VERTICES],
            scratch_vertex_plane_a: [0; MAX_VERTICES],
            scratch_vertex_plane_b: [0; MAX_VERTICES],
            scratch_vertex_count: 0,
            scratch_edge_planes: [0; MAX_VERTICES],

            seeded: false,
            failed: None,
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
        let plane_a = self.get_plane(a);
        let plane_b = self.get_plane(b);
        let plane_c = self.get_plane(c);

        let v0 = Self::intersect_two_planes(&plane_a, &plane_b, self.generator);
        let v1 = Self::intersect_two_planes(&plane_b, &plane_c, self.generator);
        let v2 = Self::intersect_two_planes(&plane_c, &plane_a, self.generator);

        // Check that all 3 vertices satisfy ALL accumulated half-spaces
        for plane_idx in 0..self.plane_count {
            if plane_idx == a || plane_idx == b || plane_idx == c {
                continue;
            }
            let plane = self.get_plane(plane_idx);
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

        // Clear vertices
        self.vertex_count = 0;

        // Check winding order - triangle normal should point towards generator
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1.cross(edge2);

        let mut min_cos = 1.0f32;
        if normal.dot(self.generator) >= 0.0 {
            // CCW winding when viewed from generator
            self.push_vertex(v0, a, b);
            min_cos = min_cos.min(self.generator.dot(v0).clamp(-1.0, 1.0) as f32);
            self.push_vertex(v1, b, c);
            min_cos = min_cos.min(self.generator.dot(v1).clamp(-1.0, 1.0) as f32);
            self.push_vertex(v2, c, a);
            min_cos = min_cos.min(self.generator.dot(v2).clamp(-1.0, 1.0) as f32);
            self.edge_planes[0] = b;
            self.edge_planes[1] = c;
            self.edge_planes[2] = a;
        } else {
            // Reverse winding
            self.push_vertex(v0, a, b);
            min_cos = min_cos.min(self.generator.dot(v0).clamp(-1.0, 1.0) as f32);
            self.push_vertex(v2, c, a);
            min_cos = min_cos.min(self.generator.dot(v2).clamp(-1.0, 1.0) as f32);
            self.push_vertex(v1, b, c);
            min_cos = min_cos.min(self.generator.dot(v1).clamp(-1.0, 1.0) as f32);
            self.edge_planes[0] = a;
            self.edge_planes[1] = c;
            self.edge_planes[2] = b;
        }

        self.cached_min_vertex_cos = min_cos;
        self.cached_min_vertex_cos_valid = true;
        true
    }

    /// Try to seed the polygon by finding a valid triplet among accumulated planes.
    fn try_seed(&mut self) -> bool {
        let n = self.plane_count;
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
    fn clip_with_plane(&mut self, plane_idx: usize) -> Result<(), CellFailure> {
        let new_plane = self.get_plane(plane_idx);
        let n = self.vertex_count;
        if n < 3 {
            return Ok(());
        }

        // Classify vertices
        let mut inside_count = 0usize;
        for i in 0..n {
            let v_pos = self.get_vertex_pos(i);
            let inside = new_plane.signed_distance(v_pos) >= -F64_EPS_CLIP;
            inside_count += inside as usize;
            self.scratch_inside[i] = inside;
        }
        if inside_count == n {
            return Ok(()); // All inside, no clipping needed
        }
        if inside_count == 0 {
            self.vertex_count = 0;
            self.failed = Some(CellFailure::ClippedAway);
            self.cached_min_vertex_cos = 1.0;
            self.cached_min_vertex_cos_valid = false;
            return Err(CellFailure::ClippedAway);
        }

        // Find entry and exit points
        let mut entry_idx: Option<usize> = None;
        let mut exit_idx: Option<usize> = None;

        for i in 0..n {
            let next = (i + 1) % n;
            let inside_i = self.scratch_inside[i];
            let inside_next = self.scratch_inside[next];
            if inside_i && !inside_next {
                exit_idx = Some(i);
            }
            if !inside_i && inside_next {
                entry_idx = Some(i);
            }
        }

        let (entry_idx, exit_idx) = match (entry_idx, exit_idx) {
            (Some(e), Some(x)) => (e, x),
            _ => return Ok(()),
        };

        // Clear scratch output buffers
        self.scratch_vertex_count = 0;
        let mut scratch_edge_count = 0usize;

        // Compute entry vertex
        let entry_edge_plane = self.edge_planes[entry_idx];
        let entry_start = self.get_vertex_pos(entry_idx);
        let entry_end = self.get_vertex_pos((entry_idx + 1) % n);
        let d_start = new_plane.signed_distance(entry_start);
        let d_end = new_plane.signed_distance(entry_end);
        let t = d_start / (d_start - d_end);
        let entry_hint = (entry_start * (1.0 - t) + entry_end * t).normalize();
        let entry_edge_gc = self.get_plane(entry_edge_plane);
        let entry_pos = Self::intersect_two_planes(&entry_edge_gc, &new_plane, entry_hint);

        // Compute exit vertex
        let exit_edge_plane = self.edge_planes[exit_idx];
        let exit_start = self.get_vertex_pos(exit_idx);
        let exit_end = self.get_vertex_pos((exit_idx + 1) % n);
        let d_start_exit = new_plane.signed_distance(exit_start);
        let d_end_exit = new_plane.signed_distance(exit_end);
        let t_exit = d_start_exit / (d_start_exit - d_end_exit);
        let exit_hint = (exit_start * (1.0 - t_exit) + exit_end * t_exit).normalize();
        let exit_edge_gc = self.get_plane(exit_edge_plane);
        let exit_pos = Self::intersect_two_planes(&exit_edge_gc, &new_plane, exit_hint);

        // Build new vertex list in scratch buffers
        self.push_scratch_vertex(entry_pos, entry_edge_plane, plane_idx)?;
        self.scratch_edge_planes[scratch_edge_count] = entry_edge_plane;
        scratch_edge_count += 1;
        let mut min_cos = self.generator.dot(entry_pos).clamp(-1.0, 1.0) as f32;

        let mut i = (entry_idx + 1) % n;
        while i != (exit_idx + 1) % n {
            let v_pos = self.get_vertex_pos(i);
            min_cos = min_cos.min(self.generator.dot(v_pos).clamp(-1.0, 1.0) as f32);
            self.push_scratch_vertex(v_pos, self.vertex_plane_a[i], self.vertex_plane_b[i])?;
            self.scratch_edge_planes[scratch_edge_count] = self.edge_planes[i];
            scratch_edge_count += 1;
            i = (i + 1) % n;
        }

        self.push_scratch_vertex(exit_pos, exit_edge_plane, plane_idx)?;
        self.scratch_edge_planes[scratch_edge_count] = plane_idx;
        min_cos = min_cos.min(self.generator.dot(exit_pos).clamp(-1.0, 1.0) as f32);

        // Copy scratch to vertices
        self.copy_scratch_to_vertices();
        self.cached_min_vertex_cos = min_cos;
        self.cached_min_vertex_cos_valid = true;
        Ok(())
    }

    /// Add a neighbor and clip the cell.
    pub fn clip(&mut self, neighbor_idx: usize, neighbor: Vec3) -> Result<(), CellFailure> {
        if self.failed.is_some() {
            return Err(self.failed.unwrap());
        }

        // Overflow bailout - if we exceed MAX_PLANES, fail
        if self.plane_count >= MAX_PLANES {
            self.failed = Some(CellFailure::TooManyPlanes);
            return Err(CellFailure::TooManyPlanes);
        }

        let n64 = DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64).normalize();

        // Skip degenerate bisectors
        let diff = self.generator - n64;
        if diff.length_squared() < (MIN_BISECTOR_DISTANCE as f64).powi(2) {
            return Ok(());
        }

        let new_plane = F64GreatCircle::bisector(self.generator, n64);
        let plane_idx = self.plane_count;
        self.plane_normals[plane_idx] = new_plane.normal;
        self.neighbor_indices[plane_idx] = neighbor_idx;
        self.plane_count += 1;

        if !self.seeded {
            if self.plane_count >= 3 {
                if self.try_seed() {
                    self.seeded = true;
                    // Clip with all non-seed planes
                    for idx in 0..self.plane_count {
                        // Skip planes that are part of current vertices
                        let is_seed_plane = (0..self.vertex_count)
                            .any(|vi| self.vertex_plane_a[vi] == idx || self.vertex_plane_b[vi] == idx);
                        if !is_seed_plane {
                            self.clip_with_plane(idx)?;
                        }
                    }
                }
            }
            return Ok(());
        }

        self.clip_with_plane(plane_idx)
    }

    /// Check if the cell has failed.
    pub fn is_failed(&self) -> bool {
        self.failed.is_some()
    }

    /// Get the failure reason, if any.
    pub fn failure(&self) -> Option<CellFailure> {
        self.failed
    }

    /// Get vertex count.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Get plane count (number of neighbors clipped).
    #[inline]
    pub fn planes_count(&self) -> usize {
        self.plane_count
    }

    /// Returns true if this neighbor has already been clipped into the cell.
    #[inline]
    pub fn has_neighbor(&self, neighbor_idx: usize) -> bool {
        self.neighbor_indices[..self.plane_count].contains(&neighbor_idx)
    }

    /// Returns an iterator over the neighbor indices that have been clipped.
    #[inline]
    pub fn neighbor_indices_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbor_indices[..self.plane_count].iter().copied()
    }

    /// Get the minimum cosine (furthest vertex angle) from generator.
    /// Returns 1.0 if no vertices.
    #[inline]
    pub fn min_vertex_cos(&self) -> f64 {
        if self.cached_min_vertex_cos_valid {
            self.cached_min_vertex_cos as f64
        } else {
            (0..self.vertex_count)
                .map(|i| self.generator.dot(self.get_vertex_pos(i)).clamp(-1.0, 1.0))
                .fold(1.0f64, f64::min)
        }
    }

    /// Check if we can terminate early based on security radius.
    pub fn can_terminate(&self, next_neighbor_cos: f32) -> bool {
        if self.vertex_count < 3 {
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
        let plane_count = self.plane_count;
        if plane_count < 3 {
            return false;
        }

        let min_score = (F64_EPS_PARALLEL * F64_EPS_PARALLEL) as f64;
        let mut best_triplet: Option<(usize, usize, usize)> = None;
        let mut best_score = 0.0f64;

        for a in 0..plane_count {
            for b in (a + 1)..plane_count {
                for c in (b + 1)..plane_count {
                    let na = self.plane_normals[a];
                    let nb = self.plane_normals[b];
                    let nc = self.plane_normals[c];
                    let ab = na.cross(nb).length_squared();
                    let bc = nb.cross(nc).length_squared();
                    let ca = nc.cross(na).length_squared();
                    let score = ab.min(bc).min(ca);
                    if score <= best_score.max(min_score) {
                        continue;
                    }

                    self.vertex_count = 0;
                    self.failed = None;
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

        self.vertex_count = 0;
        self.failed = None;
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
            if self.clip_with_plane(plane_idx).is_err() {
                return false;
            }
        }

        self.vertex_count >= 3
    }

    /// Reset the builder for a new cell.
    pub fn reset(&mut self, generator_idx: usize, generator: Vec3) {
        let gen64 = DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64).normalize();
        self.generator_idx = generator_idx;
        self.generator = gen64;
        self.plane_count = 0;
        self.vertex_count = 0;
        self.scratch_vertex_count = 0;
        self.seeded = false;
        self.failed = None;
        self.cached_min_vertex_cos = 1.0;
        self.cached_min_vertex_cos_valid = false;
    }

    /// Iterator over vertices for testing/inspection.
    /// Returns (vertex_index, position, plane_a, plane_b) for each vertex.
    pub fn vertices_iter(&self) -> impl Iterator<Item = (usize, DVec3, usize, usize)> + '_ {
        (0..self.vertex_count).map(move |i| {
            (
                i,
                self.get_vertex_pos(i),
                self.vertex_plane_a[i],
                self.vertex_plane_b[i],
            )
        })
    }

    /// Get the plane normal for a given plane index.
    pub fn plane_normal(&self, plane_idx: usize) -> DVec3 {
        self.plane_normals[plane_idx]
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
        if self.failed.is_some() || self.vertex_count < 3 {
            return Vec::new();
        }

        let g = self.generator;
        Self::debug_assert_unitish(g);
        let gen_idx = self.generator_idx as u32;

        // Pre-load all neighbor positions once (instead of per-vertex).
        let neighbor_positions: Vec<DVec3> = self.neighbor_indices[..self.plane_count]
            .iter()
            .map(|&idx| {
                let p = points[idx];
                DVec3::new(p.x as f64, p.y as f64, p.z as f64).normalize()
            })
            .collect();

        (0..self.vertex_count)
            .map(|vi| {
                let v_pos = self.get_vertex_pos(vi);
                let v_plane_a = self.vertex_plane_a[vi];
                let v_plane_b = self.vertex_plane_b[vi];
                Self::debug_assert_unit(v_pos);
                let dot_g = v_pos.dot(g);

                // Defining generators (always in support set)
                let def_a = self.neighbor_indices[v_plane_a] as u32;
                let def_b = self.neighbor_indices[v_plane_b] as u32;

                let mut min_gap = f64::INFINITY;
                let mut support_set = vec![gen_idx, def_a, def_b];

                // Check all other generators
                for (plane_idx, c) in neighbor_positions.iter().enumerate() {
                    // Skip defining planes
                    if plane_idx == v_plane_a || plane_idx == v_plane_b {
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

    /// Compute the conditioning (sin of angle between defining planes) for a vertex by index.
    fn vertex_conditioning_at(&self, vi: usize) -> f64 {
        let n_a = self.plane_normals[self.vertex_plane_a[vi]];
        let n_b = self.plane_normals[self.vertex_plane_b[vi]];
        n_a.cross(n_b).length()
    }

    /// Compute the conditioning for a vertex by index (for testing).
    pub fn vertex_conditioning_by_index(&self, idx: usize) -> f64 {
        self.vertex_conditioning_at(idx)
    }

    /// Convert f64 vertices to f32 VertexData with certified support keys.
    ///
    /// Uses slack-based certification: for each vertex, we compute the gap to all
    /// non-defining generators. The support set includes generators with gap <= eps_support.
    /// Certification passes if min_gap > eps_support + error_margin.
    ///
    /// Returns Err if certification fails.
    pub fn to_vertex_data(
        &self,
        points: &[Vec3],
        support_data: &mut Vec<u32>,
    ) -> Result<Vec<VertexData>, CellFailure> {
        let mut out = Vec::new();
        self.to_vertex_data_into(points, support_data, &mut out)?;
        Ok(out)
    }

    /// Convert f64 vertices to f32 VertexData with certified support keys, writing into `out`.
    ///
    /// This is equivalent to `to_vertex_data` but allows reusing allocations across cells.
    pub fn to_vertex_data_into(
        &self,
        points: &[Vec3],
        support_data: &mut Vec<u32>,
        out: &mut Vec<VertexData>,
    ) -> Result<(), CellFailure> {
        out.clear();
        if self.failed.is_some() || self.vertex_count < 3 {
            return Ok(());
        }

        let g = self.generator;
        Self::debug_assert_unitish(g);
        let gen_idx = self.generator_idx as u32;

        // Pre-normalize neighbor positions once into a fixed stack buffer (no heap alloc).
        let mut neighbor_positions = [DVec3::ZERO; MAX_PLANES];
        for plane_idx in 0..self.plane_count {
            let p = points[self.neighbor_indices[plane_idx]];
            neighbor_positions[plane_idx] =
                DVec3::new(p.x as f64, p.y as f64, p.z as f64).normalize();
        }

        // F64 vertex error bound (Case A from plan2.md):
        // vertex_err ≈ C * f64::EPSILON / conditioning
        // where conditioning = sin(angle between defining planes)
        //
        // Gap error = vertex_err * |G - C| (generator distance)
        // For certification: min_gap > SUPPORT_EPS_ABS + gap_error_bound
        //
        // We use C = 16 to account for multiple f64 operations (normalize, cross, etc.)
        const F64_VERTEX_ERR_FACTOR: f64 = 16.0 * f64::EPSILON;

        // Scratch reused only for rare support-set cases (near-degenerate vertices).
        let mut support_tmp: Vec<u32> = Vec::with_capacity(MAX_PLANES + 1);
        let mut support_extra = [0u32; MAX_PLANES];

        out.reserve(self.vertex_count);
        for vertex_idx in 0..self.vertex_count {
            let v_pos = self.get_vertex_pos(vertex_idx);
            Self::debug_assert_unit(v_pos);

            let pos = Vec3::new(v_pos.x as f32, v_pos.y as f32, v_pos.z as f32);
            let dot_g = v_pos.dot(g);

            let v_plane_a = self.vertex_plane_a[vertex_idx];
            let v_plane_b = self.vertex_plane_b[vertex_idx];
            let def_a = self.neighbor_indices[v_plane_a] as u32;
            let def_b = self.neighbor_indices[v_plane_b] as u32;

            let mut min_gap = f64::INFINITY;
            let mut extra_len = 0usize;

            for plane_idx in 0..self.plane_count {
                if plane_idx == v_plane_a || plane_idx == v_plane_b {
                    continue;
                }

                let dot_c = v_pos.dot(neighbor_positions[plane_idx]);
                let gap = dot_g - dot_c;

                if gap <= SUPPORT_EPS_ABS {
                    support_extra[extra_len] = self.neighbor_indices[plane_idx] as u32;
                    extra_len += 1;
                } else if gap < min_gap {
                    min_gap = gap;
                }
            }

            // Per-vertex conditioning-based error bound
            let conditioning = self.vertex_conditioning_at(vertex_idx).max(1e-6);
            let vertex_err = F64_VERTEX_ERR_FACTOR / conditioning;
            // Gap error is vertex_err * max_generator_distance
            // Conservative: use 2.0 as upper bound for |G - C| on unit sphere
            let gap_err_bound = vertex_err * 2.0;
            let cert_threshold = SUPPORT_EPS_ABS + gap_err_bound;

            if min_gap <= cert_threshold {
                return Err(CellFailure::CertificationFailed);
            }

            let key = if extra_len == 0 {
                let mut a = gen_idx;
                let mut b = def_a;
                let mut c = def_b;
                Self::sort3(&mut a, &mut b, &mut c);
                VertexKey::Triplet([a, b, c])
            } else {
                support_tmp.clear();
                support_tmp.push(gen_idx);
                support_tmp.push(def_a);
                support_tmp.push(def_b);
                support_tmp.extend_from_slice(&support_extra[..extra_len]);
                support_tmp.sort_unstable();
                support_tmp.dedup();

                if support_tmp.len() == 3 {
                    VertexKey::Triplet([support_tmp[0], support_tmp[1], support_tmp[2]])
                } else if support_tmp.len() >= 4 {
                    let start = support_data.len() as u32;
                    support_data.extend_from_slice(&support_tmp);
                    let len = u8::try_from(support_tmp.len())
                        .map_err(|_| CellFailure::CertificationFailed)?;
                    VertexKey::Support { start, len }
                } else {
                    return Err(CellFailure::CertificationFailed);
                }
            };

            out.push((key, pos));
        }

        Ok(())
    }
}
