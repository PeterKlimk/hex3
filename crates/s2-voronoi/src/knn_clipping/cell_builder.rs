//! Cell building types and algorithms for spherical Voronoi computation.

use glam::{DVec3, Vec3};

use super::constants::COINCIDENT_DOT_TOL;

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
/// Dimensionless multiplier for f64 vertex position error.
const VERTEX_ERR_FACTOR: f64 = 16.0;
/// Conditioning floor to avoid exploding error bounds on degenerate triplets.
const CONDITIONING_FLOOR: f64 = 1e-6;

/// Support-set epsilon in dot space (absolute).
///
/// This controls how aggressively we treat a vertex as "near-degenerate" (supported by >2
/// neighbor planes). Keeping this small avoids collapsing distinct vertices into the same
/// support-set key, which can both hurt performance and create per-cell duplicate indices.
const SUPPORT_EPS_ABS: f64 = 1e-12;

/// Conservative angular uncertainty (radians) used by early-termination bounds.
///
/// Even though the cell builder uses f64 arithmetic, inputs and kNN are driven by f32,
/// so termination needs an f32-scale pad. This is intended to be tight enough to allow
/// early termination on well-spaced point sets.
const TERMINATION_VERTEX_ANGLE_EPS: f64 = 8.0 * f32::EPSILON as f64;

/// Conditioning threshold for termination checks.
///
/// Termination assumes vertex directions are accurate enough to bound the true cell radius.
/// When vertices are defined by nearly-parallel planes, the intersection direction can be
/// extremely ill-conditioned; in that case we disable termination (and rely on the k schedule
/// / full-scan fallback) instead of risking a false early-out.
const TERMINATION_MIN_CONDITIONING: f64 = CONDITIONING_FLOOR;

/// Additional (conditioning-dependent) angular pad applied on top of `TERMINATION_VERTEX_ANGLE_EPS`.
///
/// This is a conservative hedge for poorly-conditioned vertices, where small input noise (f32
/// quantization from the kNN path) can move vertex directions enough to affect the bound.
const TERMINATION_CONDITIONING_PAD_FACTOR: f64 = 64.0;

/// Cap for the conditioning-dependent pad (radians).
const TERMINATION_CONDITIONING_PAD_MAX: f64 = 1e-3;

/// Conservative dot-space slack for early-termination comparisons.
///
/// This accounts for f32 dot-product rounding in the kNN path (the `next_neighbor_cos`
/// value) without inflating the angular pad via an overly pessimistic dot→angle conversion.
const TERMINATION_COS_MARGIN: f64 = 3.0 * f32::EPSILON as f64;

#[inline]
fn vertex_err(conditioning: f64) -> f64 {
    VERTEX_ERR_FACTOR * f64::EPSILON / conditioning.max(CONDITIONING_FLOOR)
}

#[inline]
fn support_cutoff(conditioning: f64) -> f64 {
    // We treat the input points as exact (they are already quantized to f32).
    // The relevant uncertainty is the numeric error in the computed vertex position,
    // which can be amplified for poorly-conditioned plane intersections.
    SUPPORT_EPS_ABS + 2.0 * vertex_err(conditioning)
}

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
    /// Cached normalized neighbor positions (f64) - computed once in clip(), reused in certification
    neighbor_positions: [DVec3; MAX_PLANES],
    /// Cached unnormalized predicate normals `n = g - h` for each plane.
    ///
    /// This avoids recomputing `g - h` in the inner certification loops.
    plane_normals_unnorm: [DVec3; MAX_PLANES],
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
    cached_min_vertex_cos: f64,
    cached_min_vertex_cos_valid: bool,
    sin_eps: f64,
    cos_eps: f64,
    termination_margin: f64,
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
    pub(crate) fn sort3_u32(a: &mut u32, b: &mut u32, c: &mut u32) {
        Self::sort3(a, b, c);
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
        let gen64 =
            DVec3::new(generator.x as f64, generator.y as f64, generator.z as f64).normalize();
        let eps_angle = TERMINATION_VERTEX_ANGLE_EPS;
        let (sin_eps, cos_eps) = eps_angle.sin_cos();
        let termination_margin = TERMINATION_COS_MARGIN;
        Self {
            generator_idx,
            generator: gen64,

            // Planes
            plane_normals: [DVec3::ZERO; MAX_PLANES],
            neighbor_indices: [0; MAX_PLANES],
            neighbor_positions: [DVec3::ZERO; MAX_PLANES],
            plane_normals_unnorm: [DVec3::ZERO; MAX_PLANES],
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
        // Use length_squared for parallel check to avoid sqrt on rejection path
        let len_sq = cross.length_squared();
        if len_sq < F64_EPS_PARALLEL * F64_EPS_PARALLEL {
            // Nearly parallel - use hint
            return hint.normalize();
        }
        let dir = cross / len_sq.sqrt();
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

        // Check that the seed triangle is not fully outside any accumulated half-space.
        for plane_idx in 0..self.plane_count {
            if plane_idx == a || plane_idx == b || plane_idx == c {
                continue;
            }
            let plane = self.get_plane(plane_idx);
            let d0 = plane.signed_distance(v0);
            if d0 >= -F64_EPS_CLIP {
                continue;
            }
            let d1 = plane.signed_distance(v1);
            if d1 >= -F64_EPS_CLIP {
                continue;
            }
            let d2 = plane.signed_distance(v2);
            if d2 < -F64_EPS_CLIP {
                return false;
            }
        }

        // Clear vertices
        self.vertex_count = 0;

        // Winding is determined by sign of inside_01 (reuse from containment check above)
        // Positive inside_01 means CCW winding when viewed from generator
        let mut min_cos = 1.0f64;
        if inside_01 > 0.0 {
            // CCW winding when viewed from generator
            self.push_vertex(v0, a, b);
            min_cos = min_cos.min(self.generator.dot(v0).clamp(-1.0, 1.0));
            self.push_vertex(v1, b, c);
            min_cos = min_cos.min(self.generator.dot(v1).clamp(-1.0, 1.0));
            self.push_vertex(v2, c, a);
            min_cos = min_cos.min(self.generator.dot(v2).clamp(-1.0, 1.0));
            self.edge_planes[0] = b;
            self.edge_planes[1] = c;
            self.edge_planes[2] = a;
        } else {
            // Reverse winding
            self.push_vertex(v0, a, b);
            min_cos = min_cos.min(self.generator.dot(v0).clamp(-1.0, 1.0));
            self.push_vertex(v2, c, a);
            min_cos = min_cos.min(self.generator.dot(v2).clamp(-1.0, 1.0));
            self.push_vertex(v1, b, c);
            min_cos = min_cos.min(self.generator.dot(v1).clamp(-1.0, 1.0));
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

        // Called incrementally as planes are appended, so each triplet is tried exactly once when
        // its highest index becomes the "newest plane".
        for i in 0..(n - 1) {
            for j in (i + 1)..(n - 1) {
                if self.try_seed_from_triplet(i, j, n - 1) {
                    return true;
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
        let mut min_cos = self.generator.dot(entry_pos).clamp(-1.0, 1.0);

        let mut i = (entry_idx + 1) % n;
        while i != (exit_idx + 1) % n {
            let v_pos = self.get_vertex_pos(i);
            min_cos = min_cos.min(self.generator.dot(v_pos).clamp(-1.0, 1.0));
            self.push_scratch_vertex(v_pos, self.vertex_plane_a[i], self.vertex_plane_b[i])?;
            self.scratch_edge_planes[scratch_edge_count] = self.edge_planes[i];
            scratch_edge_count += 1;
            i = (i + 1) % n;
        }

        self.push_scratch_vertex(exit_pos, exit_edge_plane, plane_idx)?;
        self.scratch_edge_planes[scratch_edge_count] = plane_idx;
        min_cos = min_cos.min(self.generator.dot(exit_pos).clamp(-1.0, 1.0));

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

        // Debug: catch duplicate neighbors early
        debug_assert!(
            !self.has_neighbor(neighbor_idx),
            "clip() called with duplicate neighbor {} for cell {}",
            neighbor_idx,
            self.generator_idx
        );

        // Overflow bailout - if we exceed MAX_PLANES, fail
        if self.plane_count >= MAX_PLANES {
            self.failed = Some(CellFailure::TooManyPlanes);
            return Err(CellFailure::TooManyPlanes);
        }

        let n64 = DVec3::new(neighbor.x as f64, neighbor.y as f64, neighbor.z as f64).normalize();

        // Skip coincident generators (duplicate inputs).
        let dot = self.generator.dot(n64).clamp(-1.0, 1.0);
        if (1.0 - dot) <= COINCIDENT_DOT_TOL as f64 {
            return Ok(());
        }

        let new_plane = F64GreatCircle::bisector(self.generator, n64);
        let plane_idx = self.plane_count;
        self.plane_normals[plane_idx] = new_plane.normal;
        self.neighbor_indices[plane_idx] = neighbor_idx;
        self.neighbor_positions[plane_idx] = n64; // Cache the normalized position
        self.plane_normals_unnorm[plane_idx] = self.generator - n64;
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

    #[inline]
    pub(crate) fn generator(&self) -> DVec3 {
        self.generator
    }

    #[inline]
    pub(crate) fn generator_index_u32(&self) -> u32 {
        self.generator_idx as u32
    }

    #[inline]
    pub(crate) fn vertex_pos_unit(&self, vi: usize) -> DVec3 {
        self.get_vertex_pos(vi)
    }

    #[inline]
    pub(crate) fn vertex_def_planes(&self, vi: usize) -> (usize, usize) {
        (self.vertex_plane_a[vi], self.vertex_plane_b[vi])
    }

    #[inline]
    pub(crate) fn plane_neighbor_index_u32(&self, plane_idx: usize) -> u32 {
        self.neighbor_indices[plane_idx] as u32
    }

    /// Unnormalized bisector plane normal for predicates: `n = g - h`.
    ///
    /// This avoids normalization (sqrt) and is scale-invariant for sign tests.
    #[inline]
    pub(crate) fn plane_normal_unnorm(&self, plane_idx: usize) -> DVec3 {
        self.plane_normals_unnorm[plane_idx]
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
            self.cached_min_vertex_cos
        } else {
            (0..self.vertex_count)
                .map(|i| self.generator.dot(self.get_vertex_pos(i)).clamp(-1.0, 1.0))
                .fold(1.0f64, f64::min)
        }
    }

    #[inline]
    fn min_vertex_conditioning_sq(&self) -> f64 {
        if self.vertex_count == 0 {
            return 1.0;
        }

        (0..self.vertex_count)
            .map(|vi| {
                let n_a = self.plane_normals[self.vertex_plane_a[vi]];
                let n_b = self.plane_normals[self.vertex_plane_b[vi]];
                n_a.cross(n_b).length_squared()
            })
            .fold(1.0f64, f64::min)
    }

    /// Compute the maximum allowed dot-product for any *unseen* generator, below which
    /// termination is safe.
    ///
    /// Returns `None` if the current cell geometry is too ill-conditioned to safely
    /// apply the bound.
    #[inline]
    pub fn termination_unseen_dot_threshold(&self) -> Option<f64> {
        if self.vertex_count < 3 {
            return None;
        }

        // If any vertex is too ill-conditioned, avoid termination entirely.
        let min_cond_sq = self.min_vertex_conditioning_sq();
        if !(min_cond_sq.is_finite() && min_cond_sq > 0.0) {
            return None;
        }
        let min_cond = min_cond_sq.sqrt();
        if min_cond < TERMINATION_MIN_CONDITIONING {
            return None;
        }

        let min_cos = self.min_vertex_cos();
        if min_cos <= 0.0 {
            return None;
        }

        // Bound derivation (triangle inequality on S²):
        // If every unseen generator c satisfies dist(g, c) > 2*max_v dist(g, v),
        // then no unseen generator can be closer than g anywhere in the cell.
        //
        // We pad the vertex angle by a small, conservative amount to account for
        // f32-scale input noise, and inflate further when vertex intersections are
        // poorly-conditioned.
        let extra_pad = (TERMINATION_CONDITIONING_PAD_FACTOR * (f32::EPSILON as f64) / min_cond)
            .min(TERMINATION_CONDITIONING_PAD_MAX);
        let angle_pad = TERMINATION_VERTEX_ANGLE_EPS + extra_pad;

        let (sin_pad, cos_pad) = if extra_pad == 0.0 {
            // Fast path: avoid per-call trig.
            (self.sin_eps, self.cos_eps)
        } else {
            angle_pad.sin_cos()
        };

        let sin_theta = (1.0 - min_cos * min_cos).max(0.0).sqrt();
        let cos_theta_pad = min_cos * cos_pad - sin_theta * sin_pad; // cos(theta + pad)
        let cos_2max = 2.0 * cos_theta_pad * cos_theta_pad - 1.0; // cos(2*(theta + pad))

        Some(cos_2max - self.termination_margin)
    }

    /// Check if we can terminate early based on security radius.
    pub fn can_terminate(&self, max_unseen_dot_bound: f32) -> bool {
        let Some(threshold) = self.termination_unseen_dot_threshold() else {
            return false;
        };
        (max_unseen_dot_bound as f64) < threshold
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
    pub fn compute_vertex_gaps(&self, _points: &[Vec3], eps_support: f64) -> Vec<(f64, Vec<u32>)> {
        if self.failed.is_some() || self.vertex_count < 3 {
            return Vec::new();
        }

        let g = self.generator;
        Self::debug_assert_unitish(g);
        let gen_idx = self.generator_idx as u32;

        // Use cached normalized neighbor positions (computed once in clip())
        let neighbor_positions = &self.neighbor_positions[..self.plane_count];

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
    /// Uses determinant-based certification with unnormalized plane normals. Support sets
    /// include generators whose determinant is zero (true degeneracy).
    ///
    /// Returns Err if certification fails.
    pub fn to_vertex_data(
        &self,
        _points: &[Vec3],
        support_data: &mut Vec<u32>,
    ) -> Result<Vec<VertexData>, CellFailure> {
        let mut out = Vec::new();
        self.to_vertex_data_into(_points, support_data, &mut out)?;
        Ok(out)
    }

    /// Convert f64 vertices to f32 VertexData with certified support keys, writing into `out`.
    ///
    /// This is equivalent to `to_vertex_data` but allows reusing allocations across cells.
    pub fn to_vertex_data_into(
        &self,
        _points: &[Vec3],
        support_data: &mut Vec<u32>,
        out: &mut Vec<VertexData>,
    ) -> Result<(), CellFailure> {
        super::certify::certify_to_vertex_data_into(self, support_data, out)
            .map_err(|_| CellFailure::CertificationFailed)
    }
}
