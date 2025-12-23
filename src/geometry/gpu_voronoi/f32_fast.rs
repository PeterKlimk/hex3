//! Benchmark-only f32 cell builder.
//!
//! NOT SAFE FOR PRODUCTION - no certification guarantees.
//! Purpose: measure performance ceiling of f32 clipping vs f64.
//!
//! Run with: cargo test --release f32_vs_f64 -- --nocapture

use glam::Vec3;

const FAST_K: usize = 24;
const MAX_VERTS: usize = 32;
const EPS_CLIP: f32 = 1e-7;

// Heuristic thresholds for "would bail" tracking
const SUPPORT_DANGER_ZONE: f32 = 1e-4;
const MIN_CONDITIONING: f32 = 1e-5;

// f32 termination margin accounting for compound errors:
// - Each f32 op has ~1.2e-7 relative error
// - Vertex computation: normalize + cross + lerp + normalize ≈ 6-10 ops
// - Conservative estimate: 20 * f32::EPSILON ≈ 2.4e-6 per vertex
// - Plus angular padding similar to f64's eps_cell
const F32_VERTEX_ANGLE_EPS: f32 = 1e-5;  // ~100x f32::EPSILON for safety
const F32_TERMINATION_MARGIN: f32 = 1e-5;  // Additional margin in cos space

/// Benchmark result for a single cell (no Option - always returns data)
#[derive(Debug, Clone)]
pub struct F32BenchResult {
    pub vertex_count: usize,
    pub positions: Vec<Vec3>,
    pub died: bool,
    pub termination_ok: bool,
    pub conditioning_ok: bool,
    pub support_ok: bool,
}

impl F32BenchResult {
    /// Would our heuristics have triggered fallback?
    /// Currently: died + termination + conditioning (support zone excluded for now)
    pub fn would_fallback(&self) -> bool {
        self.died || !self.termination_ok || !self.conditioning_ok
        // Note: support_ok intentionally excluded - will evaluate later
    }
}

/// Internal vertex representation during clipping
#[derive(Clone, Copy)]
struct ClipVertex {
    pos: Vec3,
    plane_a: u8,
    plane_b: u8,
}

/// Build cell with f32, recording what happened.
/// Always completes - no early bailout - so we can measure timing accurately.
pub fn bench_build_f32(gen: Vec3, neighbors: &[(usize, Vec3)]) -> F32BenchResult {
    debug_assert!(neighbors.len() >= FAST_K + 1, "need 25 neighbors");

    let mut planes: [Vec3; FAST_K] = [Vec3::ZERO; FAST_K];
    let mut dirs: [Vec3; FAST_K] = [Vec3::ZERO; FAST_K];
    let mut plane_count = 0usize;

    let mut verts: [ClipVertex; MAX_VERTS] = [ClipVertex {
        pos: Vec3::ZERO,
        plane_a: 0,
        plane_b: 0,
    }; MAX_VERTS];
    let mut vert_count: usize = 0;

    let mut died = false;

    // Build all planes and directions
    for (_neighbor_idx, neighbor_pos) in neighbors[..FAST_K].iter() {
        let diff = gen - *neighbor_pos;
        let len_sq = diff.length_squared();
        if len_sq < 1e-10 {
            continue;
        }
        let len = len_sq.sqrt();
        planes[plane_count] = diff / len;
        dirs[plane_count] = -diff / len;
        plane_count += 1;
    }

    // Seed using greedy-opposite heuristic first, then fall back to search
    let mut seed_triplet: Option<(u8, u8, u8)> = None;
    if plane_count >= 3 {
        // Try greedy-opposite heuristic first (95% success rate)
        let (ha, hb, hc) = heuristic_greedy_opposite(&dirs, plane_count);
        let p3 = [planes[ha], planes[hb], planes[hc]];
        if let Some((v0, v1, v2)) = seed_from_three_planes(gen, &p3) {
            let mut valid = true;
            for p in 0..plane_count {
                if p == ha || p == hb || p == hc {
                    continue;
                }
                let d0 = planes[p].dot(v0);
                let d1 = planes[p].dot(v1);
                let d2 = planes[p].dot(v2);
                if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                    valid = false;
                    break;
                }
            }
            if valid {
                verts[0] = ClipVertex { pos: v0, plane_a: ha as u8, plane_b: hb as u8 };
                verts[1] = ClipVertex { pos: v1, plane_a: hb as u8, plane_b: hc as u8 };
                verts[2] = ClipVertex { pos: v2, plane_a: hc as u8, plane_b: ha as u8 };
                vert_count = 3;
                seed_triplet = Some((ha as u8, hb as u8, hc as u8));
            }
        }

        // Fall back to search if heuristic failed
        if seed_triplet.is_none() {
            'outer: for a in 0..plane_count.min(6) {
                for b in (a + 1)..plane_count.min(8) {
                    for c in (b + 1)..plane_count.min(10) {
                        if a == ha && b == hb && c == hc {
                            continue;
                        }
                        let p3 = [planes[a], planes[b], planes[c]];
                        if let Some((v0, v1, v2)) = seed_from_three_planes(gen, &p3) {
                            let mut valid = true;
                            for p in 0..plane_count {
                                if p == a || p == b || p == c {
                                    continue;
                                }
                                let d0 = planes[p].dot(v0);
                                let d1 = planes[p].dot(v1);
                                let d2 = planes[p].dot(v2);
                                if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                                    valid = false;
                                    break;
                                }
                            }
                            if valid {
                                verts[0] = ClipVertex { pos: v0, plane_a: a as u8, plane_b: b as u8 };
                                verts[1] = ClipVertex { pos: v1, plane_a: b as u8, plane_b: c as u8 };
                                verts[2] = ClipVertex { pos: v2, plane_a: c as u8, plane_b: a as u8 };
                                vert_count = 3;
                                seed_triplet = Some((a as u8, b as u8, c as u8));
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }

    if seed_triplet.is_none() {
        died = true;
    }

    // Clip with all non-seed planes
    if let Some((sa, sb, sc)) = seed_triplet {
        for p in 0..plane_count {
            if p == sa as usize || p == sb as usize || p == sc as usize {
                continue;
            }
            match clip_polygon(&mut verts, vert_count, planes[p], p as u8) {
                Some(new_count) => vert_count = new_count,
                None => {
                    died = true;
                    vert_count = 0;
                    break;
                }
            }
        }
    }

    // Collect final positions
    let positions: Vec<Vec3> = verts[..vert_count].iter().map(|v| v.pos).collect();

    // Check 1: Termination with 25th neighbor
    let termination_ok = if !died && vert_count >= 3 {
        let (_, neighbor_25_pos) = neighbors[FAST_K];
        let worst_cos = gen.dot(neighbor_25_pos);
        let min_vert_cos = positions.iter().map(|p| gen.dot(*p)).fold(1.0f32, f32::min);
        can_terminate_f32(min_vert_cos, worst_cos)
    } else {
        false
    };

    // Check 2: Conditioning (cross product of defining planes)
    let conditioning_ok = if !died && vert_count >= 3 {
        verts[..vert_count].iter().all(|v| {
            if v.plane_a == v.plane_b {
                return true; // Seed vertex, skip check
            }
            let na = planes.get(v.plane_a as usize).copied().unwrap_or(Vec3::Z);
            let nb = planes.get(v.plane_b as usize).copied().unwrap_or(Vec3::Z);
            na.cross(nb).length() >= MIN_CONDITIONING
        })
    } else {
        false
    };

    // Check 3: Support danger zone (any non-defining plane too close)
    let support_ok = if !died && vert_count >= 3 {
        verts[..vert_count].iter().all(|v| {
            for p in 0..plane_count {
                if p == v.plane_a as usize || p == v.plane_b as usize {
                    continue;
                }
                let dist = planes[p].dot(v.pos).abs();
                if dist < SUPPORT_DANGER_ZONE {
                    return false;
                }
            }
            true
        })
    } else {
        false
    };

    F32BenchResult {
        vertex_count: vert_count,
        positions,
        died,
        termination_ok,
        conditioning_ok,
        support_ok,
    }
}

/// Compute the intersection of two planes on the unit sphere.
/// Returns the point on the generator's side.
fn intersect_two_planes(p1: Vec3, p2: Vec3, gen: Vec3) -> Option<Vec3> {
    let cross = p1.cross(p2);
    let len = cross.length();
    if len < 1e-6 {
        return None; // Nearly parallel planes
    }
    let dir = cross / len;
    // Choose sign based on generator hint
    if dir.dot(gen) >= 0.0 {
        Some(dir)
    } else {
        Some(-dir)
    }
}

/// Seed a triangle from three bisector planes.
/// Returns three vertices (intersections of plane pairs) if valid.
fn seed_from_three_planes(gen: Vec3, planes: &[Vec3]) -> Option<(Vec3, Vec3, Vec3)> {
    if planes.len() < 3 {
        return None;
    }

    // Compute vertices: intersection of each pair of planes
    let v0 = intersect_two_planes(planes[0], planes[1], gen)?;
    let v1 = intersect_two_planes(planes[1], planes[2], gen)?;
    let v2 = intersect_two_planes(planes[2], planes[0], gen)?;

    // Check that all vertices are on the correct side of all planes
    for (i, &plane) in planes.iter().enumerate() {
        let d0 = plane.dot(v0);
        let d1 = plane.dot(v1);
        let d2 = plane.dot(v2);

        // Each vertex should be on positive side of planes it's not defined by
        if i != 0 && i != 1 && d0 < -EPS_CLIP {
            return None;
        }
        if i != 1 && i != 2 && d1 < -EPS_CLIP {
            return None;
        }
        if i != 2 && i != 0 && d2 < -EPS_CLIP {
            return None;
        }
    }

    // Check that generator is inside the triangle (winding)
    let edge01 = v0.cross(v1);
    let edge12 = v1.cross(v2);
    let edge20 = v2.cross(v0);

    let d01 = edge01.dot(gen);
    let d12 = edge12.dot(gen);
    let d20 = edge20.dot(gen);

    // All same sign = generator inside triangle
    let all_pos = d01 > 0.0 && d12 > 0.0 && d20 > 0.0;
    let all_neg = d01 < 0.0 && d12 < 0.0 && d20 < 0.0;

    if !all_pos && !all_neg {
        return None;
    }

    // Return in correct winding order (CCW when viewed from generator)
    if all_pos {
        Some((v0, v1, v2))
    } else {
        Some((v0, v2, v1))
    }
}

fn clip_polygon(
    verts: &mut [ClipVertex; MAX_VERTS],
    count: usize,
    plane: Vec3,
    plane_idx: u8,
) -> Option<usize> {
    if count < 3 {
        return None;
    }

    // Classify vertices
    let mut inside = [false; MAX_VERTS];
    let mut inside_count = 0;
    for i in 0..count {
        inside[i] = plane.dot(verts[i].pos) >= -EPS_CLIP;
        inside_count += inside[i] as usize;
    }

    if inside_count == count {
        return Some(count); // All inside, no clipping needed
    }
    if inside_count == 0 {
        return None; // All outside = dead cell
    }

    // Sutherland-Hodgman clipping
    let mut out: [ClipVertex; MAX_VERTS] = [ClipVertex {
        pos: Vec3::ZERO,
        plane_a: 0,
        plane_b: 0,
    }; MAX_VERTS];
    let mut out_count = 0;

    for i in 0..count {
        let j = (i + 1) % count;
        let curr = verts[i];
        let next = verts[j];

        if inside[i] {
            out[out_count] = curr;
            out_count += 1;
        }

        if inside[i] != inside[j] {
            // Edge crosses plane - compute intersection
            let di = plane.dot(curr.pos);
            let dj = plane.dot(next.pos);
            let t = di / (di - dj);
            let new_pos = curr.pos.lerp(next.pos, t).normalize();

            // Determine edge plane for the new vertex
            let edge_plane = if inside[i] { curr.plane_b } else { next.plane_a };

            out[out_count] = ClipVertex {
                pos: new_pos,
                plane_a: edge_plane,
                plane_b: plane_idx,
            };
            out_count += 1;

            if out_count >= MAX_VERTS {
                return None; // Overflow
            }
        }
    }

    verts[..out_count].copy_from_slice(&out[..out_count]);
    Some(out_count)
}

fn can_terminate_f32(min_vert_cos: f32, worst_neighbor_cos: f32) -> bool {
    // Adapted from f64's can_terminate with f32-appropriate margins.
    //
    // Logic: if furthest vertex is at angle θ from generator, any neighbor
    // at angle > 2*(θ + eps) cannot affect that vertex (where eps accounts
    // for vertex position uncertainty from f32 arithmetic).

    if min_vert_cos <= 0.0 {
        return false;
    }

    // Compute sin(theta) from cos(theta)
    let sin_theta = (1.0 - min_vert_cos * min_vert_cos).max(0.0).sqrt();

    // Precompute sin/cos of the angular epsilon
    let (sin_eps, cos_eps) = F32_VERTEX_ANGLE_EPS.sin_cos();

    // cos(theta + eps) = cos(theta)*cos(eps) - sin(theta)*sin(eps)
    let cos_theta_eps = min_vert_cos * cos_eps - sin_theta * sin_eps;

    // cos(2 * (theta + eps)) = 2*cos²(theta + eps) - 1
    let cos_2max = 2.0 * cos_theta_eps * cos_theta_eps - 1.0;

    // Neighbor must be beyond this angle (with additional margin)
    worst_neighbor_cos < cos_2max - F32_TERMINATION_MARGIN
}

// ============================================================================
// SoA (Structure of Arrays) implementation for better cache/SIMD potential
// ============================================================================

/// SoA vertex buffer - x, y, z in separate arrays for SIMD-friendly access
struct SoaVertices {
    x: [f32; MAX_VERTS],
    y: [f32; MAX_VERTS],
    z: [f32; MAX_VERTS],
    plane_a: [u8; MAX_VERTS],
    plane_b: [u8; MAX_VERTS],
    count: usize,
}

impl SoaVertices {
    #[inline]
    fn new() -> Self {
        Self {
            x: [0.0; MAX_VERTS],
            y: [0.0; MAX_VERTS],
            z: [0.0; MAX_VERTS],
            plane_a: [0; MAX_VERTS],
            plane_b: [0; MAX_VERTS],
            count: 0,
        }
    }

    #[inline]
    fn push(&mut self, pos: Vec3, pa: u8, pb: u8) {
        let i = self.count;
        self.x[i] = pos.x;
        self.y[i] = pos.y;
        self.z[i] = pos.z;
        self.plane_a[i] = pa;
        self.plane_b[i] = pb;
        self.count += 1;
    }

    #[inline]
    fn get_pos(&self, i: usize) -> Vec3 {
        Vec3::new(self.x[i], self.y[i], self.z[i])
    }

    #[inline]
    fn set_pos(&mut self, i: usize, pos: Vec3) {
        self.x[i] = pos.x;
        self.y[i] = pos.y;
        self.z[i] = pos.z;
    }
}

/// SoA version of the f32 cell builder
pub fn bench_build_f32_soa(gen: Vec3, neighbors: &[(usize, Vec3)]) -> F32BenchResult {
    debug_assert!(neighbors.len() >= FAST_K + 1, "need 25 neighbors");

    // Build planes and directions
    let mut planes_x: [f32; FAST_K] = [0.0; FAST_K];
    let mut planes_y: [f32; FAST_K] = [0.0; FAST_K];
    let mut planes_z: [f32; FAST_K] = [0.0; FAST_K];
    let mut dirs: [Vec3; FAST_K] = [Vec3::ZERO; FAST_K];
    let mut plane_count = 0usize;

    for (_neighbor_idx, neighbor_pos) in neighbors[..FAST_K].iter() {
        let diff = gen - *neighbor_pos;
        let len_sq = diff.length_squared();
        if len_sq < 1e-10 {
            continue;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        planes_x[plane_count] = diff.x * inv_len;
        planes_y[plane_count] = diff.y * inv_len;
        planes_z[plane_count] = diff.z * inv_len;
        dirs[plane_count] = Vec3::new(-diff.x * inv_len, -diff.y * inv_len, -diff.z * inv_len);
        plane_count += 1;
    }

    let mut verts = SoaVertices::new();
    let mut died = false;

    // Seed using greedy-opposite heuristic first, then fall back to search
    let mut seed_triplet: Option<(u8, u8, u8)> = None;
    if plane_count >= 3 {
        // Try greedy-opposite heuristic first (95% success rate)
        let (ha, hb, hc) = heuristic_greedy_opposite(&dirs, plane_count);
        let p0 = Vec3::new(planes_x[ha], planes_y[ha], planes_z[ha]);
        let p1 = Vec3::new(planes_x[hb], planes_y[hb], planes_z[hb]);
        let p2 = Vec3::new(planes_x[hc], planes_y[hc], planes_z[hc]);

        if let Some((v0, v1, v2)) = seed_from_three_planes(gen, &[p0, p1, p2]) {
            let mut valid = true;
            for p in 0..plane_count {
                if p == ha || p == hb || p == hc {
                    continue;
                }
                let d0 = planes_x[p] * v0.x + planes_y[p] * v0.y + planes_z[p] * v0.z;
                let d1 = planes_x[p] * v1.x + planes_y[p] * v1.y + planes_z[p] * v1.z;
                let d2 = planes_x[p] * v2.x + planes_y[p] * v2.y + planes_z[p] * v2.z;
                if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                    valid = false;
                    break;
                }
            }
            if valid {
                verts.push(v0, ha as u8, hb as u8);
                verts.push(v1, hb as u8, hc as u8);
                verts.push(v2, hc as u8, ha as u8);
                seed_triplet = Some((ha as u8, hb as u8, hc as u8));
            }
        }

        // Fall back to search if heuristic failed
        if seed_triplet.is_none() {
            'outer: for a in 0..plane_count.min(6) {
                for b in (a + 1)..plane_count.min(8) {
                    for c in (b + 1)..plane_count.min(10) {
                        if a == ha && b == hb && c == hc {
                            continue;
                        }
                        let p0 = Vec3::new(planes_x[a], planes_y[a], planes_z[a]);
                        let p1 = Vec3::new(planes_x[b], planes_y[b], planes_z[b]);
                        let p2 = Vec3::new(planes_x[c], planes_y[c], planes_z[c]);

                        if let Some((v0, v1, v2)) = seed_from_three_planes(gen, &[p0, p1, p2]) {
                            let mut valid = true;
                            for p in 0..plane_count {
                                if p == a || p == b || p == c {
                                    continue;
                                }
                                let d0 = planes_x[p] * v0.x + planes_y[p] * v0.y + planes_z[p] * v0.z;
                                let d1 = planes_x[p] * v1.x + planes_y[p] * v1.y + planes_z[p] * v1.z;
                                let d2 = planes_x[p] * v2.x + planes_y[p] * v2.y + planes_z[p] * v2.z;
                                if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                                    valid = false;
                                    break;
                                }
                            }
                            if valid {
                                verts.push(v0, a as u8, b as u8);
                                verts.push(v1, b as u8, c as u8);
                                verts.push(v2, c as u8, a as u8);
                                seed_triplet = Some((a as u8, b as u8, c as u8));
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }

    if seed_triplet.is_none() {
        died = true;
    }

    // Clip with all non-seed planes
    if let Some((sa, sb, sc)) = seed_triplet {
        for p in 0..plane_count {
            if p == sa as usize || p == sb as usize || p == sc as usize {
                continue;
            }
            let plane = Vec3::new(planes_x[p], planes_y[p], planes_z[p]);
            if !clip_polygon_soa(&mut verts, plane, p as u8) {
                died = true;
                verts.count = 0;
                break;
            }
        }
    }

    // Collect final positions
    let positions: Vec<Vec3> = (0..verts.count).map(|i| verts.get_pos(i)).collect();

    // Termination check
    let termination_ok = if !died && verts.count >= 3 {
        let (_, neighbor_25_pos) = neighbors[FAST_K];
        let worst_cos = gen.dot(neighbor_25_pos);
        let min_vert_cos = positions.iter().map(|p| gen.dot(*p)).fold(1.0f32, f32::min);
        can_terminate_f32(min_vert_cos, worst_cos)
    } else {
        false
    };

    // Conditioning check
    let conditioning_ok = if !died && verts.count >= 3 {
        (0..verts.count).all(|i| {
            let pa = verts.plane_a[i] as usize;
            let pb = verts.plane_b[i] as usize;
            if pa == pb {
                return true;
            }
            let na = Vec3::new(planes_x[pa], planes_y[pa], planes_z[pa]);
            let nb = Vec3::new(planes_x[pb], planes_y[pb], planes_z[pb]);
            na.cross(nb).length() >= MIN_CONDITIONING
        })
    } else {
        false
    };

    // Support zone check
    let support_ok = if !died && verts.count >= 3 {
        (0..verts.count).all(|vi| {
            let vpos = verts.get_pos(vi);
            let pa = verts.plane_a[vi] as usize;
            let pb = verts.plane_b[vi] as usize;
            for p in 0..plane_count {
                if p == pa || p == pb {
                    continue;
                }
                let dist = (planes_x[p] * vpos.x + planes_y[p] * vpos.y + planes_z[p] * vpos.z).abs();
                if dist < SUPPORT_DANGER_ZONE {
                    return false;
                }
            }
            true
        })
    } else {
        false
    };

    F32BenchResult {
        vertex_count: verts.count,
        positions,
        died,
        termination_ok,
        conditioning_ok,
        support_ok,
    }
}

/// Minimal f32 SoA - just returns (vertex_count, died) for fair timing comparison with f64 SoA
/// Uses SAME tuple-based plane storage as f64 SoA for truly apples-to-apples comparison
pub fn bench_build_f32_soa_minimal(gen: Vec3, neighbors: &[(usize, Vec3)]) -> (usize, bool) {
    debug_assert!(neighbors.len() >= FAST_K + 1, "need 25 neighbors");

    let gen32 = (gen.x, gen.y, gen.z);

    // Build all planes and directions upfront (SAME structure as f64 SoA)
    let mut planes: [(f32, f32, f32); FAST_K] = [(0.0, 0.0, 0.0); FAST_K];
    let mut dirs: [Vec3; FAST_K] = [Vec3::ZERO; FAST_K];  // f32 dirs for heuristic
    let mut plane_count = 0usize;

    for (_neighbor_idx, neighbor_pos) in neighbors[..FAST_K].iter() {
        let diff = (
            gen32.0 - neighbor_pos.x,
            gen32.1 - neighbor_pos.y,
            gen32.2 - neighbor_pos.z,
        );
        let len_sq = diff.0 * diff.0 + diff.1 * diff.1 + diff.2 * diff.2;
        if len_sq < 1e-10 {
            continue;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        planes[plane_count] = (diff.0 * inv_len, diff.1 * inv_len, diff.2 * inv_len);
        dirs[plane_count] = Vec3::new(-diff.0 * inv_len, -diff.1 * inv_len, -diff.2 * inv_len);
        plane_count += 1;
    }

    let mut verts = SoaVertices::new();
    let mut died = false;

    // Seed using greedy-opposite heuristic first (SAME as f64 SoA)
    let mut seed_triplet: Option<(u8, u8, u8)> = None;
    if plane_count >= 3 {
        let (ha, hb, hc) = heuristic_greedy_opposite(&dirs, plane_count);
        let p0 = Vec3::new(planes[ha].0, planes[ha].1, planes[ha].2);
        let p1 = Vec3::new(planes[hb].0, planes[hb].1, planes[hb].2);
        let p2 = Vec3::new(planes[hc].0, planes[hc].1, planes[hc].2);

        if let Some((v0, v1, v2)) = seed_from_three_planes(gen, &[p0, p1, p2]) {
            let mut valid = true;
            for p in 0..plane_count {
                if p == ha || p == hb || p == hc {
                    continue;
                }
                let d0 = planes[p].0 * v0.x + planes[p].1 * v0.y + planes[p].2 * v0.z;
                let d1 = planes[p].0 * v1.x + planes[p].1 * v1.y + planes[p].2 * v1.z;
                let d2 = planes[p].0 * v2.x + planes[p].1 * v2.y + planes[p].2 * v2.z;
                if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                    valid = false;
                    break;
                }
            }
            if valid {
                verts.push(v0, ha as u8, hb as u8);
                verts.push(v1, hb as u8, hc as u8);
                verts.push(v2, hc as u8, ha as u8);
                seed_triplet = Some((ha as u8, hb as u8, hc as u8));
            }
        }

        // Fall back to search
        if seed_triplet.is_none() {
            'outer: for a in 0..plane_count.min(6) {
                for b in (a + 1)..plane_count.min(8) {
                    for c in (b + 1)..plane_count.min(10) {
                        if a == ha && b == hb && c == hc {
                            continue;
                        }
                        let p0 = Vec3::new(planes[a].0, planes[a].1, planes[a].2);
                        let p1 = Vec3::new(planes[b].0, planes[b].1, planes[b].2);
                        let p2 = Vec3::new(planes[c].0, planes[c].1, planes[c].2);

                        if let Some((v0, v1, v2)) = seed_from_three_planes(gen, &[p0, p1, p2]) {
                            let mut valid = true;
                            for p in 0..plane_count {
                                if p == a || p == b || p == c {
                                    continue;
                                }
                                let d0 = planes[p].0 * v0.x + planes[p].1 * v0.y + planes[p].2 * v0.z;
                                let d1 = planes[p].0 * v1.x + planes[p].1 * v1.y + planes[p].2 * v1.z;
                                let d2 = planes[p].0 * v2.x + planes[p].1 * v2.y + planes[p].2 * v2.z;
                                if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                                    valid = false;
                                    break;
                                }
                            }
                            if valid {
                                verts.push(v0, a as u8, b as u8);
                                verts.push(v1, b as u8, c as u8);
                                verts.push(v2, c as u8, a as u8);
                                seed_triplet = Some((a as u8, b as u8, c as u8));
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }

    if seed_triplet.is_none() {
        died = true;
    }

    // Clip with all non-seed planes
    if let Some((sa, sb, sc)) = seed_triplet {
        for p in 0..plane_count {
            if p == sa as usize || p == sb as usize || p == sc as usize {
                continue;
            }
            let plane = Vec3::new(planes[p].0, planes[p].1, planes[p].2);
            if !clip_polygon_soa(&mut verts, plane, p as u8) {
                died = true;
                verts.count = 0;
                break;
            }
        }
    }

    (verts.count, died)
}

/// SoA version of clip_polygon - returns false if cell dies
fn clip_polygon_soa(verts: &mut SoaVertices, plane: Vec3, plane_idx: u8) -> bool {
    let count = verts.count;
    if count < 3 {
        return false;
    }

    // Classify all vertices - SoA enables potential SIMD here
    let mut inside = [false; MAX_VERTS];
    let mut inside_count = 0usize;

    // Manual SoA dot products (compiler can vectorize this)
    for i in 0..count {
        let dot = plane.x * verts.x[i] + plane.y * verts.y[i] + plane.z * verts.z[i];
        inside[i] = dot >= -EPS_CLIP;
        inside_count += inside[i] as usize;
    }

    if inside_count == count {
        return true; // All inside
    }
    if inside_count == 0 {
        return false; // All outside = dead
    }

    // Sutherland-Hodgman into temp buffer
    let mut out = SoaVertices::new();

    for i in 0..count {
        let j = (i + 1) % count;

        if inside[i] {
            out.x[out.count] = verts.x[i];
            out.y[out.count] = verts.y[i];
            out.z[out.count] = verts.z[i];
            out.plane_a[out.count] = verts.plane_a[i];
            out.plane_b[out.count] = verts.plane_b[i];
            out.count += 1;
        }

        if inside[i] != inside[j] {
            // Edge crosses plane
            let di = plane.x * verts.x[i] + plane.y * verts.y[i] + plane.z * verts.z[i];
            let dj = plane.x * verts.x[j] + plane.y * verts.y[j] + plane.z * verts.z[j];
            let t = di / (di - dj);

            // Lerp
            let nx = verts.x[i] + t * (verts.x[j] - verts.x[i]);
            let ny = verts.y[i] + t * (verts.y[j] - verts.y[i]);
            let nz = verts.z[i] + t * (verts.z[j] - verts.z[i]);

            // Normalize
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let inv_len = 1.0 / len;

            let edge_plane = if inside[i] { verts.plane_b[i] } else { verts.plane_a[j] };

            out.x[out.count] = nx * inv_len;
            out.y[out.count] = ny * inv_len;
            out.z[out.count] = nz * inv_len;
            out.plane_a[out.count] = edge_plane;
            out.plane_b[out.count] = plane_idx;
            out.count += 1;

            if out.count >= MAX_VERTS {
                return false;
            }
        }
    }

    // Copy back
    verts.count = out.count;
    verts.x[..out.count].copy_from_slice(&out.x[..out.count]);
    verts.y[..out.count].copy_from_slice(&out.y[..out.count]);
    verts.z[..out.count].copy_from_slice(&out.z[..out.count]);
    verts.plane_a[..out.count].copy_from_slice(&out.plane_a[..out.count]);
    verts.plane_b[..out.count].copy_from_slice(&out.plane_b[..out.count]);

    true
}

// ============================================================================
// f64 SoA implementation - same algorithm, same data structures, just f64
// ============================================================================

const F64_EPS_CLIP: f64 = EPS_CLIP as f64;  // Same epsilon as f32, just f64 type

/// SoA vertex buffer for f64
struct SoaVertices64 {
    x: [f64; MAX_VERTS],
    y: [f64; MAX_VERTS],
    z: [f64; MAX_VERTS],
    plane_a: [u8; MAX_VERTS],
    plane_b: [u8; MAX_VERTS],
    count: usize,
}

impl SoaVertices64 {
    #[inline]
    fn new() -> Self {
        Self {
            x: [0.0; MAX_VERTS],
            y: [0.0; MAX_VERTS],
            z: [0.0; MAX_VERTS],
            plane_a: [0; MAX_VERTS],
            plane_b: [0; MAX_VERTS],
            count: 0,
        }
    }

    #[inline]
    fn push(&mut self, x: f64, y: f64, z: f64, pa: u8, pb: u8) {
        let i = self.count;
        self.x[i] = x;
        self.y[i] = y;
        self.z[i] = z;
        self.plane_a[i] = pa;
        self.plane_b[i] = pb;
        self.count += 1;
    }
}

/// Seed from three planes - f64 version
fn seed_from_three_planes_f64(
    gen: (f64, f64, f64),
    p0: (f64, f64, f64),
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
) -> Option<((f64, f64, f64), (f64, f64, f64), (f64, f64, f64))> {
    // v0 = intersection of p0 and p1
    let cross01 = (
        p0.1 * p1.2 - p0.2 * p1.1,
        p0.2 * p1.0 - p0.0 * p1.2,
        p0.0 * p1.1 - p0.1 * p1.0,
    );
    let len01 = (cross01.0 * cross01.0 + cross01.1 * cross01.1 + cross01.2 * cross01.2).sqrt();
    if len01 < 1e-10 {
        return None;
    }
    let inv01 = 1.0 / len01;
    let dir01 = (cross01.0 * inv01, cross01.1 * inv01, cross01.2 * inv01);
    let dot01 = dir01.0 * gen.0 + dir01.1 * gen.1 + dir01.2 * gen.2;
    let v0 = if dot01 >= 0.0 { dir01 } else { (-dir01.0, -dir01.1, -dir01.2) };

    // v1 = intersection of p1 and p2
    let cross12 = (
        p1.1 * p2.2 - p1.2 * p2.1,
        p1.2 * p2.0 - p1.0 * p2.2,
        p1.0 * p2.1 - p1.1 * p2.0,
    );
    let len12 = (cross12.0 * cross12.0 + cross12.1 * cross12.1 + cross12.2 * cross12.2).sqrt();
    if len12 < 1e-10 {
        return None;
    }
    let inv12 = 1.0 / len12;
    let dir12 = (cross12.0 * inv12, cross12.1 * inv12, cross12.2 * inv12);
    let dot12 = dir12.0 * gen.0 + dir12.1 * gen.1 + dir12.2 * gen.2;
    let v1 = if dot12 >= 0.0 { dir12 } else { (-dir12.0, -dir12.1, -dir12.2) };

    // v2 = intersection of p2 and p0
    let cross20 = (
        p2.1 * p0.2 - p2.2 * p0.1,
        p2.2 * p0.0 - p2.0 * p0.2,
        p2.0 * p0.1 - p2.1 * p0.0,
    );
    let len20 = (cross20.0 * cross20.0 + cross20.1 * cross20.1 + cross20.2 * cross20.2).sqrt();
    if len20 < 1e-10 {
        return None;
    }
    let inv20 = 1.0 / len20;
    let dir20 = (cross20.0 * inv20, cross20.1 * inv20, cross20.2 * inv20);
    let dot20 = dir20.0 * gen.0 + dir20.1 * gen.1 + dir20.2 * gen.2;
    let v2 = if dot20 >= 0.0 { dir20 } else { (-dir20.0, -dir20.1, -dir20.2) };

    // Check winding (CCW when viewed from generator)
    let e1 = (v1.0 - v0.0, v1.1 - v0.1, v1.2 - v0.2);
    let e2 = (v2.0 - v0.0, v2.1 - v0.1, v2.2 - v0.2);
    let cross = (
        e1.1 * e2.2 - e1.2 * e2.1,
        e1.2 * e2.0 - e1.0 * e2.2,
        e1.0 * e2.1 - e1.1 * e2.0,
    );
    let winding = cross.0 * gen.0 + cross.1 * gen.1 + cross.2 * gen.2;
    if winding < 0.0 {
        return None;
    }

    Some((v0, v1, v2))
}

/// SoA clip for f64
fn clip_polygon_soa_f64(
    verts: &mut SoaVertices64,
    plane: (f64, f64, f64),
    plane_idx: u8,
) -> bool {
    let count = verts.count;
    if count < 3 {
        return false;
    }

    let mut inside = [false; MAX_VERTS];
    let mut inside_count = 0usize;

    for i in 0..count {
        let dot = plane.0 * verts.x[i] + plane.1 * verts.y[i] + plane.2 * verts.z[i];
        inside[i] = dot >= -F64_EPS_CLIP;
        inside_count += inside[i] as usize;
    }

    if inside_count == count {
        return true;
    }
    if inside_count == 0 {
        return false;
    }

    let mut out = SoaVertices64::new();

    for i in 0..count {
        let j = (i + 1) % count;

        if inside[i] {
            out.x[out.count] = verts.x[i];
            out.y[out.count] = verts.y[i];
            out.z[out.count] = verts.z[i];
            out.plane_a[out.count] = verts.plane_a[i];
            out.plane_b[out.count] = verts.plane_b[i];
            out.count += 1;
        }

        if inside[i] != inside[j] {
            let di = plane.0 * verts.x[i] + plane.1 * verts.y[i] + plane.2 * verts.z[i];
            let dj = plane.0 * verts.x[j] + plane.1 * verts.y[j] + plane.2 * verts.z[j];
            let t = di / (di - dj);

            let nx = verts.x[i] + t * (verts.x[j] - verts.x[i]);
            let ny = verts.y[i] + t * (verts.y[j] - verts.y[i]);
            let nz = verts.z[i] + t * (verts.z[j] - verts.z[i]);

            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let inv_len = 1.0 / len;

            let edge_plane = if inside[i] { verts.plane_b[i] } else { verts.plane_a[j] };

            out.x[out.count] = nx * inv_len;
            out.y[out.count] = ny * inv_len;
            out.z[out.count] = nz * inv_len;
            out.plane_a[out.count] = edge_plane;
            out.plane_b[out.count] = plane_idx;
            out.count += 1;

            if out.count >= MAX_VERTS {
                return false;
            }
        }
    }

    verts.count = out.count;
    verts.x[..out.count].copy_from_slice(&out.x[..out.count]);
    verts.y[..out.count].copy_from_slice(&out.y[..out.count]);
    verts.z[..out.count].copy_from_slice(&out.z[..out.count]);
    verts.plane_a[..out.count].copy_from_slice(&out.plane_a[..out.count]);
    verts.plane_b[..out.count].copy_from_slice(&out.plane_b[..out.count]);

    true
}

/// f64 SoA benchmark - EXACT same algorithm as f32 SoA, just with f64 precision
pub fn bench_build_f64_soa(gen: Vec3, neighbors: &[(usize, Vec3)]) -> (usize, bool) {
    debug_assert!(neighbors.len() >= FAST_K + 1, "need 25 neighbors");

    let gen64 = (gen.x as f64, gen.y as f64, gen.z as f64);

    // Build all planes and directions upfront (same as f32 SoA)
    let mut planes: [(f64, f64, f64); FAST_K] = [(0.0, 0.0, 0.0); FAST_K];
    let mut dirs: [Vec3; FAST_K] = [Vec3::ZERO; FAST_K];  // f32 dirs for heuristic
    let mut plane_count = 0usize;

    for (_neighbor_idx, neighbor_pos) in neighbors[..FAST_K].iter() {
        let diff = (
            gen64.0 - neighbor_pos.x as f64,
            gen64.1 - neighbor_pos.y as f64,
            gen64.2 - neighbor_pos.z as f64,
        );
        let len_sq = diff.0 * diff.0 + diff.1 * diff.1 + diff.2 * diff.2;
        if len_sq < 1e-10 {
            continue;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        planes[plane_count] = (diff.0 * inv_len, diff.1 * inv_len, diff.2 * inv_len);
        // f32 dirs for heuristic (just for picking indices)
        dirs[plane_count] = Vec3::new(
            (-diff.0 * inv_len) as f32,
            (-diff.1 * inv_len) as f32,
            (-diff.2 * inv_len) as f32,
        );
        plane_count += 1;
    }

    let mut verts = SoaVertices64::new();
    let mut died = false;

    // Seed using greedy-opposite heuristic first, then fall back to search
    // (SAME as f32 SoA)
    let mut seed_triplet: Option<(u8, u8, u8)> = None;
    if plane_count >= 3 {
        // Try greedy-opposite heuristic first
        let (ha, hb, hc) = heuristic_greedy_opposite(&dirs, plane_count);

        if let Some((v0, v1, v2)) = seed_from_three_planes_f64(
            gen64,
            planes[ha],
            planes[hb],
            planes[hc],
        ) {
            let mut valid = true;
            for p in 0..plane_count {
                if p == ha || p == hb || p == hc {
                    continue;
                }
                let d0 = planes[p].0 * v0.0 + planes[p].1 * v0.1 + planes[p].2 * v0.2;
                let d1 = planes[p].0 * v1.0 + planes[p].1 * v1.1 + planes[p].2 * v1.2;
                let d2 = planes[p].0 * v2.0 + planes[p].1 * v2.1 + planes[p].2 * v2.2;
                if d0 < -F64_EPS_CLIP && d1 < -F64_EPS_CLIP && d2 < -F64_EPS_CLIP {
                    valid = false;
                    break;
                }
            }
            if valid {
                verts.push(v0.0, v0.1, v0.2, ha as u8, hb as u8);
                verts.push(v1.0, v1.1, v1.2, hb as u8, hc as u8);
                verts.push(v2.0, v2.1, v2.2, hc as u8, ha as u8);
                seed_triplet = Some((ha as u8, hb as u8, hc as u8));
            }
        }

        // Fall back to search if heuristic failed (same bounds as f32 SoA)
        if seed_triplet.is_none() {
            'outer: for a in 0..plane_count.min(6) {
                for b in (a + 1)..plane_count.min(8) {
                    for c in (b + 1)..plane_count.min(10) {
                        if a == ha && b == hb && c == hc {
                            continue;
                        }

                        if let Some((v0, v1, v2)) = seed_from_three_planes_f64(
                            gen64,
                            planes[a],
                            planes[b],
                            planes[c],
                        ) {
                            let mut valid = true;
                            for p in 0..plane_count {
                                if p == a || p == b || p == c {
                                    continue;
                                }
                                let d0 = planes[p].0 * v0.0 + planes[p].1 * v0.1 + planes[p].2 * v0.2;
                                let d1 = planes[p].0 * v1.0 + planes[p].1 * v1.1 + planes[p].2 * v1.2;
                                let d2 = planes[p].0 * v2.0 + planes[p].1 * v2.1 + planes[p].2 * v2.2;
                                if d0 < -F64_EPS_CLIP && d1 < -F64_EPS_CLIP && d2 < -F64_EPS_CLIP {
                                    valid = false;
                                    break;
                                }
                            }
                            if valid {
                                verts.push(v0.0, v0.1, v0.2, a as u8, b as u8);
                                verts.push(v1.0, v1.1, v1.2, b as u8, c as u8);
                                verts.push(v2.0, v2.1, v2.2, c as u8, a as u8);
                                seed_triplet = Some((a as u8, b as u8, c as u8));
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
    }

    if seed_triplet.is_none() {
        died = true;
    }

    // Clip with all non-seed planes (same as f32 SoA)
    if let Some((sa, sb, sc)) = seed_triplet {
        for p in 0..plane_count {
            if p == sa as usize || p == sb as usize || p == sc as usize {
                continue;
            }
            if !clip_polygon_soa_f64(&mut verts, planes[p], p as u8) {
                died = true;
                verts.count = 0;
                break;
            }
        }
    }

    (verts.count, died)
}

/// Seed triplet search statistics
#[derive(Default, Clone, Copy)]
pub struct SeedTripletStats {
    /// First triplet (0,1,2) worked
    pub first_success: u64,
    /// First triplet failed, but found another
    pub search_success: u64,
    /// No valid triplet found (cell died)
    pub search_failed: u64,
    /// First triplet failed due to plane intersection
    pub first_fail_intersection: u64,
    /// First triplet failed due to winding/containment
    pub first_fail_winding: u64,
    /// First triplet failed due to other plane clipping all vertices
    pub first_fail_other_clips: u64,
    /// Heuristic 1 (axis-aligned) would have worked
    pub heuristic_axis_success: u64,
    /// Heuristic 2 (greedy opposite) would have worked
    pub heuristic_greedy_success: u64,
}

/// Result of trying to seed from a specific triplet
#[derive(Debug, Clone, Copy)]
pub enum TripletResult {
    Success,
    FailIntersection,  // Planes too parallel
    FailWinding,       // Generator not inside triangle
    FailOtherClips,    // Another plane clips all 3 vertices
}

/// Test whether the first triplet (0,1,2) would succeed
fn test_first_triplet(gen: Vec3, planes: &[Vec3], plane_count: usize) -> TripletResult {
    if plane_count < 3 {
        return TripletResult::FailIntersection;
    }

    let p3 = [planes[0], planes[1], planes[2]];

    // Try to compute the seed triangle
    match seed_from_three_planes(gen, &p3) {
        None => {
            // Failed - determine if it was intersection or winding
            // Try computing raw intersections to see if planes are parallel
            let v0 = intersect_two_planes(planes[0], planes[1], gen);
            let v1 = intersect_two_planes(planes[1], planes[2], gen);
            let v2 = intersect_two_planes(planes[2], planes[0], gen);

            if v0.is_none() || v1.is_none() || v2.is_none() {
                TripletResult::FailIntersection
            } else {
                TripletResult::FailWinding
            }
        }
        Some((v0, v1, v2)) => {
            // Seed computed, but check if other planes clip all 3 vertices
            for p in 3..plane_count {
                let d0 = planes[p].dot(v0);
                let d1 = planes[p].dot(v1);
                let d2 = planes[p].dot(v2);
                if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                    return TripletResult::FailOtherClips;
                }
            }
            TripletResult::Success
        }
    }
}

/// Heuristic 1: Axis-aligned selection
/// Pick neighbors with extreme projections on different axes
fn heuristic_axis_aligned(dirs: &[Vec3], count: usize) -> (usize, usize, usize) {
    if count < 3 {
        return (0, 1.min(count - 1), 2.min(count - 1));
    }

    // Find max X, min X, and max |Y|
    let mut max_x_idx = 0;
    let mut min_x_idx = 0;
    let mut max_y_idx = 0;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_x = f32::INFINITY;
    let mut max_abs_y = 0.0f32;

    for i in 0..count {
        let d = dirs[i];
        if d.x > max_x {
            max_x = d.x;
            max_x_idx = i;
        }
        if d.x < min_x {
            min_x = d.x;
            min_x_idx = i;
        }
        if d.y.abs() > max_abs_y {
            max_abs_y = d.y.abs();
            max_y_idx = i;
        }
    }

    // Handle collisions - if max_y_idx equals one of the X indices, find another
    if max_y_idx == max_x_idx || max_y_idx == min_x_idx {
        // Find max |Z| instead
        let mut max_abs_z = 0.0f32;
        let mut max_z_idx = 0;
        for i in 0..count {
            if i == max_x_idx || i == min_x_idx {
                continue;
            }
            if dirs[i].z.abs() > max_abs_z {
                max_abs_z = dirs[i].z.abs();
                max_z_idx = i;
            }
        }
        max_y_idx = max_z_idx;
    }

    // Sort indices to get consistent ordering
    let mut indices = [max_x_idx, min_x_idx, max_y_idx];
    indices.sort_unstable();
    (indices[0], indices[1], indices[2])
}

/// Heuristic 2: Greedy opposite selection
/// Start with 0, find most opposite, then most orthogonal
fn heuristic_greedy_opposite(dirs: &[Vec3], count: usize) -> (usize, usize, usize) {
    if count < 3 {
        return (0, 1.min(count - 1), 2.min(count - 1));
    }

    let a = 0; // Start with first (closest) neighbor
    let dir_a = dirs[a];

    // Find most opposite to a (min dot product)
    let mut b = 1;
    let mut min_dot = f32::INFINITY;
    for i in 1..count {
        let dot = dir_a.dot(dirs[i]);
        if dot < min_dot {
            min_dot = dot;
            b = i;
        }
    }

    // Find most orthogonal to plane of a and b
    // The plane normal is dir_a × dir_b
    let plane_normal = dir_a.cross(dirs[b]);
    let plane_len = plane_normal.length();

    let mut c = if b == 1 { 2 } else { 1 };
    if plane_len > 1e-6 {
        let plane_normal = plane_normal / plane_len;
        let mut max_orth = 0.0f32;
        for i in 1..count {
            if i == b {
                continue;
            }
            let orth = dirs[i].dot(plane_normal).abs();
            if orth > max_orth {
                max_orth = orth;
                c = i;
            }
        }
    }

    // Sort indices
    let mut indices = [a, b, c];
    indices.sort_unstable();
    (indices[0], indices[1], indices[2])
}

/// Test a specific triplet and return if it's valid
fn test_triplet_valid(
    gen: Vec3,
    planes: &[Vec3],
    plane_count: usize,
    a: usize,
    b: usize,
    c: usize,
) -> bool {
    let p3 = [planes[a], planes[b], planes[c]];
    match seed_from_three_planes(gen, &p3) {
        None => false,
        Some((v0, v1, v2)) => {
            // Check if other planes clip all vertices
            for p in 0..plane_count {
                if p == a || p == b || p == c {
                    continue;
                }
                let d0 = planes[p].dot(v0);
                let d1 = planes[p].dot(v1);
                let d2 = planes[p].dot(v2);
                if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                    return false;
                }
            }
            true
        }
    }
}

/// Analyze seed triplet search behavior for a cell
pub fn analyze_seed_triplet(gen: Vec3, neighbors: &[(usize, Vec3)]) -> SeedTripletStats {
    let mut stats = SeedTripletStats::default();

    // Build planes and directions
    let mut planes: [Vec3; FAST_K] = [Vec3::ZERO; FAST_K];
    let mut dirs: [Vec3; FAST_K] = [Vec3::ZERO; FAST_K];
    let mut plane_count = 0usize;

    for (_neighbor_idx, neighbor_pos) in neighbors[..FAST_K].iter() {
        let diff = gen - *neighbor_pos;
        let len_sq = diff.length_squared();
        if len_sq < 1e-10 {
            continue;
        }
        let len = len_sq.sqrt();
        planes[plane_count] = diff / len;
        dirs[plane_count] = -diff / len; // Direction FROM gen TO neighbor
        plane_count += 1;
    }

    if plane_count < 3 {
        stats.search_failed = 1;
        return stats;
    }

    // Test first triplet
    let first_result = test_first_triplet(gen, &planes, plane_count);

    match first_result {
        TripletResult::Success => {
            stats.first_success = 1;
            // Also check heuristics for comparison
            let (a, b, c) = heuristic_axis_aligned(&dirs, plane_count);
            if test_triplet_valid(gen, &planes, plane_count, a, b, c) {
                stats.heuristic_axis_success = 1;
            }
            let (a, b, c) = heuristic_greedy_opposite(&dirs, plane_count);
            if test_triplet_valid(gen, &planes, plane_count, a, b, c) {
                stats.heuristic_greedy_success = 1;
            }
            return stats;
        }
        TripletResult::FailIntersection => stats.first_fail_intersection = 1,
        TripletResult::FailWinding => stats.first_fail_winding = 1,
        TripletResult::FailOtherClips => stats.first_fail_other_clips = 1,
    }

    // First triplet failed - test heuristics
    let (a, b, c) = heuristic_axis_aligned(&dirs, plane_count);
    if test_triplet_valid(gen, &planes, plane_count, a, b, c) {
        stats.heuristic_axis_success = 1;
    }

    let (a, b, c) = heuristic_greedy_opposite(&dirs, plane_count);
    if test_triplet_valid(gen, &planes, plane_count, a, b, c) {
        stats.heuristic_greedy_success = 1;
    }

    // Try the full search
    let mut found = false;
    'outer: for a in 0..plane_count.min(6) {
        for b in (a + 1)..plane_count.min(8) {
            for c in (b + 1)..plane_count.min(10) {
                if a == 0 && b == 1 && c == 2 {
                    continue; // Skip first triplet, already tested
                }

                let p3 = [planes[a], planes[b], planes[c]];
                if let Some((v0, v1, v2)) = seed_from_three_planes(gen, &p3) {
                    // Check if other planes clip all vertices
                    let mut valid = true;
                    for p in 0..plane_count {
                        if p == a || p == b || p == c {
                            continue;
                        }
                        let d0 = planes[p].dot(v0);
                        let d1 = planes[p].dot(v1);
                        let d2 = planes[p].dot(v2);
                        if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        found = true;
                        break 'outer;
                    }
                }
            }
        }
    }

    if found {
        stats.search_success = 1;
    } else {
        stats.search_failed = 1;
    }

    stats
}

// ============================================================================
/// Incremental with early termination.
/// After each clip, checks if remaining neighbors are too far to affect the cell.
pub fn bench_build_f32_incremental_early_term(gen: Vec3, neighbors: &[(usize, Vec3)]) -> F32BenchResult {
    debug_assert!(neighbors.len() >= FAST_K + 1, "need 25 neighbors");

    let mut planes: [Vec3; FAST_K] = [Vec3::ZERO; FAST_K];
    let mut plane_count = 0usize;

    let mut verts = SoaVertices::new();
    let mut died = false;
    let mut seeded = false;

    for (neighbor_i, (_neighbor_idx, neighbor_pos)) in neighbors[..FAST_K].iter().enumerate() {
        let diff = gen - *neighbor_pos;
        let len_sq = diff.length_squared();
        if len_sq < 1e-10 {
            continue;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        let plane = Vec3::new(diff.x * inv_len, diff.y * inv_len, diff.z * inv_len);

        let plane_idx = plane_count;
        planes[plane_count] = plane;
        plane_count += 1;

        if !seeded {
            if plane_count >= 3 {
                // Sliding window: try (n-3, n-2, n-1) - the 3 most recent planes
                let a = plane_count - 3;
                let b = plane_count - 2;
                let c = plane_count - 1;

                let p0 = planes[a];
                let p1 = planes[b];
                let p2 = planes[c];

                if let Some((v0, v1, v2)) = seed_from_three_planes(gen, &[p0, p1, p2]) {
                    // Check validity against existing planes
                    let mut valid = true;
                    for p in 0..plane_count {
                        if p == a || p == b || p == c {
                            continue;
                        }
                        let d0 = planes[p].dot(v0);
                        let d1 = planes[p].dot(v1);
                        let d2 = planes[p].dot(v2);
                        if d0 < -EPS_CLIP && d1 < -EPS_CLIP && d2 < -EPS_CLIP {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        verts.push(v0, a as u8, b as u8);
                        verts.push(v1, b as u8, c as u8);
                        verts.push(v2, c as u8, a as u8);
                        seeded = true;

                        // Clip with non-seed planes we've seen so far
                        for p in 0..plane_count {
                            if p == a || p == b || p == c {
                                continue;
                            }
                            if !clip_polygon_soa(&mut verts, planes[p], p as u8) {
                                died = true;
                                verts.count = 0;
                                break;
                            }
                        }
                    }
                }
            }
        } else if !died {
            // Clip with this plane
            if !clip_polygon_soa(&mut verts, plane, plane_idx as u8) {
                died = true;
                verts.count = 0;
            } else if verts.count >= 3 {
                // Check early termination: can the next neighbor affect us?
                // Look at neighbor_i + 1 (the next neighbor in distance order)
                if neighbor_i + 1 < neighbors.len() {
                    let (_, next_pos) = neighbors[neighbor_i + 1];
                    let next_cos = gen.dot(next_pos);

                    // Find min vertex cos (furthest vertex from generator)
                    let mut min_vert_cos = 1.0f32;
                    for vi in 0..verts.count {
                        let vpos = verts.get_pos(vi);
                        let cos = gen.dot(vpos);
                        if cos < min_vert_cos {
                            min_vert_cos = cos;
                        }
                    }

                    // Can we terminate?
                    if can_terminate_f32(min_vert_cos, next_cos) {
                        break;
                    }
                }
            }
        }
    }

    if !seeded {
        died = true;
    }

    let positions: Vec<Vec3> = (0..verts.count).map(|i| verts.get_pos(i)).collect();

    let termination_ok = if !died && verts.count >= 3 {
        let (_, neighbor_25_pos) = neighbors[FAST_K];
        let worst_cos = gen.dot(neighbor_25_pos);
        let min_vert_cos = positions.iter().map(|p| gen.dot(*p)).fold(1.0f32, f32::min);
        can_terminate_f32(min_vert_cos, worst_cos)
    } else {
        false
    };

    let conditioning_ok = if !died && verts.count >= 3 {
        (0..verts.count).all(|i| {
            let pa = verts.plane_a[i] as usize;
            let pb = verts.plane_b[i] as usize;
            if pa == pb { return true; }
            planes[pa].cross(planes[pb]).length() >= MIN_CONDITIONING
        })
    } else {
        false
    };

    let support_ok = if !died && verts.count >= 3 {
        (0..verts.count).all(|vi| {
            let vpos = verts.get_pos(vi);
            let pa = verts.plane_a[vi] as usize;
            let pb = verts.plane_b[vi] as usize;
            for p in 0..plane_count {
                if p == pa || p == pb { continue; }
                if planes[p].dot(vpos).abs() < SUPPORT_DANGER_ZONE { return false; }
            }
            true
        })
    } else {
        false
    };

    F32BenchResult {
        vertex_count: verts.count,
        positions,
        died,
        termination_ok,
        conditioning_ok,
        support_ok,
    }
}

/// Statistics about termination margins
#[derive(Default)]
struct TerminationStats {
    count: usize,
    sum_margin: f64,
    min_margin: f32,
    max_margin: f32,
    passed: usize,
}

impl TerminationStats {
    fn record(&mut self, min_vert_cos: f32, worst_neighbor_cos: f32) {
        if min_vert_cos <= 0.0 {
            return;
        }

        // Use the same formula as can_terminate_f32
        let sin_theta = (1.0 - min_vert_cos * min_vert_cos).max(0.0).sqrt();
        let (sin_eps, cos_eps) = F32_VERTEX_ANGLE_EPS.sin_cos();
        let cos_theta_eps = min_vert_cos * cos_eps - sin_theta * sin_eps;
        let cos_2max = 2.0 * cos_theta_eps * cos_theta_eps - 1.0;

        // margin = how much room we have (positive = neighbor is far enough)
        let margin = cos_2max - worst_neighbor_cos;

        self.count += 1;
        self.sum_margin += margin as f64;
        if self.count == 1 {
            self.min_margin = margin;
            self.max_margin = margin;
        } else {
            self.min_margin = self.min_margin.min(margin);
            self.max_margin = self.max_margin.max(margin);
        }
        if margin > F32_TERMINATION_MARGIN {
            self.passed += 1;
        }
    }

    fn report(&self) {
        if self.count == 0 {
            return;
        }
        let avg = self.sum_margin / self.count as f64;
        println!(
            "    termination margin: min={:.6} avg={:.6} max={:.6} (need >{:.6} to pass)",
            self.min_margin, avg, self.max_margin, F32_TERMINATION_MARGIN
        );
        println!(
            "    would pass w/ f32 margin: {} ({:.2}%)",
            self.passed,
            self.passed as f64 / self.count as f64 * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::gpu_voronoi::cell_builder::F64CellBuilder;
    use crate::geometry::gpu_voronoi::CubeMapGridKnn;
    use crate::geometry::fibonacci_sphere_points_with_rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    fn get_neighbors(knn: &CubeMapGridKnn, points: &[Vec3], idx: usize, k: usize) -> Vec<(usize, Vec3)> {
        let mut scratch = knn.make_scratch();
        let mut neighbors = Vec::new();
        knn.knn_resumable_into(points[idx], idx, k, k, &mut scratch, &mut neighbors);
        neighbors.iter().map(|&j| (j, points[j])).collect()
    }

    #[test]
    fn test_f32_vs_f64_timing() {
        for &n in &[10_000, 100_000, 1_000_000] {
            run_comparison(n);
        }
    }

    fn run_comparison(n: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let points = fibonacci_sphere_points_with_rng(n, 0.01, &mut rng);
        let knn = CubeMapGridKnn::new(&points);

        println!("\n=== f32 vs f64 Cell Builder Benchmark ===");
        println!("Points: {}", n);

        // Pre-fetch all neighbor lists (so KNN isn't in the timing)
        let all_neighbors: Vec<Vec<(usize, Vec3)>> = (0..n)
            .map(|i| get_neighbors(&knn, &points, i, FAST_K + 1))
            .collect();

        // Benchmark f32 AoS (current)
        let t0 = Instant::now();
        let f32_aos_results: Vec<F32BenchResult> = (0..n)
            .map(|i| bench_build_f32(points[i], &all_neighbors[i]))
            .collect();
        let f32_aos_time = t0.elapsed();

        // Benchmark f32 SoA (batch)
        let t0 = Instant::now();
        let f32_soa_results: Vec<F32BenchResult> = (0..n)
            .map(|i| bench_build_f32_soa(points[i], &all_neighbors[i]))
            .collect();
        let f32_soa_time = t0.elapsed();

        // Benchmark f32 Incremental with early termination
        let t0 = Instant::now();
        let f32_incr_early_results: Vec<F32BenchResult> = (0..n)
            .map(|i| bench_build_f32_incremental_early_term(points[i], &all_neighbors[i]))
            .collect();
        let f32_incr_early_time = t0.elapsed();

        // Benchmark f32 SoA minimal (fair comparison with f64 SoA - same return type)
        let t0 = Instant::now();
        let f32_soa_minimal_results: Vec<(usize, bool)> = (0..n)
            .map(|i| bench_build_f32_soa_minimal(points[i], &all_neighbors[i]))
            .collect();
        let f32_soa_minimal_time = t0.elapsed();

        // Benchmark f64 (production - Vec/AoS)
        let t0 = Instant::now();
        let f64_results: Vec<usize> = (0..n)
            .map(|i| {
                let mut builder = F64CellBuilder::new(i, points[i]);
                for &(idx, pos) in &all_neighbors[i][..FAST_K] {
                    builder.clip(idx, pos);
                }
                builder.vertex_count()
            })
            .collect();
        let f64_time = t0.elapsed();

        // Benchmark f64 SoA (same algorithm as f32, stack arrays)
        let t0 = Instant::now();
        let f64_soa_results: Vec<(usize, bool)> = (0..n)
            .map(|i| bench_build_f64_soa(points[i], &all_neighbors[i]))
            .collect();
        let f64_soa_time = t0.elapsed();

        // Collect death rates
        let aos_died = f32_aos_results.iter().filter(|r| r.died).count();
        let soa_died = f32_soa_results.iter().filter(|r| r.died).count();
        let soa_minimal_died = f32_soa_minimal_results.iter().filter(|r| r.1).count();
        let incr_early_died = f32_incr_early_results.iter().filter(|r| r.died).count();
        let f64_soa_died = f64_soa_results.iter().filter(|r| r.1).count();

        println!("\n{:<22} {:>10} {:>8} {:>10}", "Method", "Time (ms)", "Speedup", "Died");
        println!("{:-<52}", "");
        println!("{:<22} {:>10.2} {:>8.2}x {:>9.2}%",
            "f32 AoS (batch)",
            f32_aos_time.as_secs_f64() * 1000.0,
            f64_time.as_secs_f64() / f32_aos_time.as_secs_f64(),
            aos_died as f64 / n as f64 * 100.0
        );
        println!("{:<22} {:>10.2} {:>8.2}x {:>9.2}%",
            "f32 SoA (batch)",
            f32_soa_time.as_secs_f64() * 1000.0,
            f64_time.as_secs_f64() / f32_soa_time.as_secs_f64(),
            soa_died as f64 / n as f64 * 100.0
        );
        println!("{:<22} {:>10.2} {:>8.2}x {:>9.2}%",
            "f32 Incr (early-term)",
            f32_incr_early_time.as_secs_f64() * 1000.0,
            f64_time.as_secs_f64() / f32_incr_early_time.as_secs_f64(),
            incr_early_died as f64 / n as f64 * 100.0
        );
        println!("{:<22} {:>10.2} {:>8.2}x {:>9.2}%",
            "f32 SoA (minimal)",
            f32_soa_minimal_time.as_secs_f64() * 1000.0,
            f64_time.as_secs_f64() / f32_soa_minimal_time.as_secs_f64(),
            soa_minimal_died as f64 / n as f64 * 100.0
        );
        println!("{:<22} {:>10.2} {:>8.2}x {:>9.2}%",
            "f64 SoA (stack)",
            f64_soa_time.as_secs_f64() * 1000.0,
            f64_time.as_secs_f64() / f64_soa_time.as_secs_f64(),
            f64_soa_died as f64 / n as f64 * 100.0
        );
        println!("{:<22} {:>10.2} {:>8}",
            "f64 (prod)",
            f64_time.as_secs_f64() * 1000.0,
            "1.00x"
        );

        // Direct f32 vs f64 SoA comparison (apples-to-apples)
        println!("\n>>> f32 SoA minimal vs f64 SoA: {:.2}x",
            f64_soa_time.as_secs_f64() / f32_soa_minimal_time.as_secs_f64());

        // Incremental correctness check (vs SoA)
        let incr_early_soa_match = (0..n).filter(|&i| f32_incr_early_results[i].vertex_count == f32_soa_results[i].vertex_count).count();

        println!("\nIncremental vs SoA vertex match:");
        println!("  Incr (early-term): {:.2}%", incr_early_soa_match as f64 / n as f64 * 100.0);

        // Accuracy stats (SoA batch vs f64)
        let mut exact_match = 0;
        let mut f32_more = 0;
        let mut f32_less = 0;

        for i in 0..n {
            let f32_count = f32_soa_results[i].vertex_count;
            let f64_count = f64_results[i];
            if f32_count == f64_count {
                exact_match += 1;
            } else if f32_count > f64_count {
                f32_more += 1;
            } else {
                f32_less += 1;
            }
        }

        println!("\nVertex count (SoA batch vs f64):");
        println!(
            "  exact match: {} ({:.2}%)",
            exact_match,
            exact_match as f64 / n as f64 * 100.0
        );
        println!(
            "  f32 > f64:   {} ({:.2}%)",
            f32_more,
            f32_more as f64 / n as f64 * 100.0
        );
        println!(
            "  f32 < f64:   {} ({:.2}%)",
            f32_less,
            f32_less as f64 / n as f64 * 100.0
        );

        // Heuristic check stats (use SoA batch results)
        let died = f32_soa_results.iter().filter(|r| r.died).count();
        let term_fail = f32_soa_results.iter().filter(|r| !r.termination_ok).count();
        let cond_fail = f32_soa_results.iter().filter(|r| !r.conditioning_ok).count();
        let supp_fail = f32_soa_results.iter().filter(|r| !r.support_ok).count();
        let would_fb = f32_soa_results.iter().filter(|r| r.would_fallback()).count();

        // Compute termination margin statistics
        let mut term_stats = TerminationStats::default();
        for i in 0..n {
            if !f32_soa_results[i].died && f32_soa_results[i].vertex_count >= 3 {
                let min_vert_cos = f32_soa_results[i].positions.iter()
                    .map(|p| points[i].dot(*p))
                    .fold(1.0f32, f32::min);
                let (_, neighbor_25_pos) = all_neighbors[i][FAST_K];
                let worst_neighbor_cos = points[i].dot(neighbor_25_pos);
                term_stats.record(min_vert_cos, worst_neighbor_cos);
            }
        }

        // Analyze seed triplet behavior
        let mut total_seed_stats = SeedTripletStats::default();
        for i in 0..n {
            let s = analyze_seed_triplet(points[i], &all_neighbors[i]);
            total_seed_stats.first_success += s.first_success;
            total_seed_stats.search_success += s.search_success;
            total_seed_stats.search_failed += s.search_failed;
            total_seed_stats.first_fail_intersection += s.first_fail_intersection;
            total_seed_stats.first_fail_winding += s.first_fail_winding;
            total_seed_stats.first_fail_other_clips += s.first_fail_other_clips;
            total_seed_stats.heuristic_axis_success += s.heuristic_axis_success;
            total_seed_stats.heuristic_greedy_success += s.heuristic_greedy_success;
        }

        println!("\nHeuristic failure rates (SoA batch):");
        println!("  died:           {} ({:.2}%)", died, died as f64 / n as f64 * 100.0);
        println!(
            "  termination:    {} ({:.2}%)",
            term_fail,
            term_fail as f64 / n as f64 * 100.0
        );
        println!(
            "  conditioning:   {} ({:.2}%)",
            cond_fail,
            cond_fail as f64 / n as f64 * 100.0
        );
        println!(
            "  support zone:   {} ({:.2}%)",
            supp_fail,
            supp_fail as f64 / n as f64 * 100.0
        );
        println!(
            "  would fallback: {} ({:.2}%)",
            would_fb,
            would_fb as f64 / n as f64 * 100.0
        );
        term_stats.report();

        // Report seed triplet statistics
        println!("\nSeed triplet analysis:");
        let first_ok = total_seed_stats.first_success;
        let first_fail = total_seed_stats.search_success + total_seed_stats.search_failed;
        println!(
            "  first triplet (0,1,2) OK: {} ({:.2}%)",
            first_ok,
            first_ok as f64 / n as f64 * 100.0
        );
        println!(
            "  first triplet failed:     {} ({:.2}%)",
            first_fail,
            first_fail as f64 / n as f64 * 100.0
        );
        if first_fail > 0 {
            println!("    - intersection fail: {} ({:.2}%)",
                total_seed_stats.first_fail_intersection,
                total_seed_stats.first_fail_intersection as f64 / first_fail as f64 * 100.0);
            println!("    - winding fail:      {} ({:.2}%)",
                total_seed_stats.first_fail_winding,
                total_seed_stats.first_fail_winding as f64 / first_fail as f64 * 100.0);
            println!("    - other clips fail:  {} ({:.2}%)",
                total_seed_stats.first_fail_other_clips,
                total_seed_stats.first_fail_other_clips as f64 / first_fail as f64 * 100.0);
            println!("  search found another:     {} ({:.2}% of failures)",
                total_seed_stats.search_success,
                total_seed_stats.search_success as f64 / first_fail as f64 * 100.0);
            println!("  search failed (died):     {} ({:.2}% of failures)",
                total_seed_stats.search_failed,
                total_seed_stats.search_failed as f64 / first_fail as f64 * 100.0);
        }

        // Report heuristic success rates
        let total_heur_axis = total_seed_stats.heuristic_axis_success;
        let total_heur_greedy = total_seed_stats.heuristic_greedy_success;
        println!("\nHeuristic success rates (all cells):");
        println!(
            "  axis-aligned:    {} ({:.2}%)",
            total_heur_axis,
            total_heur_axis as f64 / n as f64 * 100.0
        );
        println!(
            "  greedy-opposite: {} ({:.2}%)",
            total_heur_greedy,
            total_heur_greedy as f64 / n as f64 * 100.0
        );
        println!(
            "  first (0,1,2):   {} ({:.2}%)",
            first_ok,
            first_ok as f64 / n as f64 * 100.0
        );
    }

    /// Micro-benchmark to understand f32 vs f64 performance difference
    #[test]
    fn test_f32_vs_f64_micro() {
        use glam::DVec3;
        use std::hint::black_box;

        const N: usize = 10_000_000;

        // Generate test data
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let points_f32: Vec<Vec3> = (0..N)
            .map(|_| {
                use rand::Rng;
                Vec3::new(
                    rng.gen::<f32>() * 2.0 - 1.0,
                    rng.gen::<f32>() * 2.0 - 1.0,
                    rng.gen::<f32>() * 2.0 - 1.0,
                ).normalize()
            })
            .collect();

        let points_f64: Vec<DVec3> = points_f32
            .iter()
            .map(|p| DVec3::new(p.x as f64, p.y as f64, p.z as f64))
            .collect();

        println!("\n=== f32 vs f64 Micro-Benchmark ({} ops) ===\n", N);

        // 1. Dot products
        let t0 = Instant::now();
        let mut sum_f32 = 0.0f32;
        for i in 0..N-1 {
            sum_f32 += points_f32[i].dot(points_f32[i + 1]);
        }
        let dot_f32_time = t0.elapsed();
        black_box(sum_f32);

        let t0 = Instant::now();
        let mut sum_f64 = 0.0f64;
        for i in 0..N-1 {
            sum_f64 += points_f64[i].dot(points_f64[i + 1]);
        }
        let dot_f64_time = t0.elapsed();
        black_box(sum_f64);

        println!("Dot product:");
        println!("  f32: {:>8.2} ms", dot_f32_time.as_secs_f64() * 1000.0);
        println!("  f64: {:>8.2} ms", dot_f64_time.as_secs_f64() * 1000.0);
        println!("  ratio: {:.2}x", dot_f64_time.as_secs_f64() / dot_f32_time.as_secs_f64());

        // 2. Normalize
        let t0 = Instant::now();
        let mut acc_f32 = Vec3::ZERO;
        for i in 0..N {
            acc_f32 += (points_f32[i] * 1.001).normalize();
        }
        let norm_f32_time = t0.elapsed();
        black_box(acc_f32);

        let t0 = Instant::now();
        let mut acc_f64 = DVec3::ZERO;
        for i in 0..N {
            acc_f64 += (points_f64[i] * 1.001).normalize();
        }
        let norm_f64_time = t0.elapsed();
        black_box(acc_f64);

        println!("\nNormalize:");
        println!("  f32: {:>8.2} ms", norm_f32_time.as_secs_f64() * 1000.0);
        println!("  f64: {:>8.2} ms", norm_f64_time.as_secs_f64() * 1000.0);
        println!("  ratio: {:.2}x", norm_f64_time.as_secs_f64() / norm_f32_time.as_secs_f64());

        // 3. Cross product
        let t0 = Instant::now();
        let mut acc_f32 = Vec3::ZERO;
        for i in 0..N-1 {
            acc_f32 += points_f32[i].cross(points_f32[i + 1]);
        }
        let cross_f32_time = t0.elapsed();
        black_box(acc_f32);

        let t0 = Instant::now();
        let mut acc_f64 = DVec3::ZERO;
        for i in 0..N-1 {
            acc_f64 += points_f64[i].cross(points_f64[i + 1]);
        }
        let cross_f64_time = t0.elapsed();
        black_box(acc_f64);

        println!("\nCross product:");
        println!("  f32: {:>8.2} ms", cross_f32_time.as_secs_f64() * 1000.0);
        println!("  f64: {:>8.2} ms", cross_f64_time.as_secs_f64() * 1000.0);
        println!("  ratio: {:.2}x", cross_f64_time.as_secs_f64() / cross_f32_time.as_secs_f64());

        // 4. Memory read throughput
        let t0 = Instant::now();
        let mut sum_f32 = Vec3::ZERO;
        for p in &points_f32 {
            sum_f32 += *p;
        }
        let read_f32_time = t0.elapsed();
        black_box(sum_f32);

        let t0 = Instant::now();
        let mut sum_f64 = DVec3::ZERO;
        for p in &points_f64 {
            sum_f64 += *p;
        }
        let read_f64_time = t0.elapsed();
        black_box(sum_f64);

        println!("\nMemory read (sum all):");
        println!("  f32: {:>8.2} ms  ({:.1} GB/s)",
            read_f32_time.as_secs_f64() * 1000.0,
            (N * 12) as f64 / read_f32_time.as_secs_f64() / 1e9);
        println!("  f64: {:>8.2} ms  ({:.1} GB/s)",
            read_f64_time.as_secs_f64() * 1000.0,
            (N * 24) as f64 / read_f64_time.as_secs_f64() / 1e9);
        println!("  ratio: {:.2}x", read_f64_time.as_secs_f64() / read_f32_time.as_secs_f64());

        // 5. Combined: normalize + dot (like plane computation)
        let t0 = Instant::now();
        let mut sum_f32 = 0.0f32;
        for i in 0..N-1 {
            let diff = points_f32[i] - points_f32[i + 1];
            let plane = diff.normalize();
            sum_f32 += plane.dot(points_f32[i]);
        }
        let combined_f32_time = t0.elapsed();
        black_box(sum_f32);

        let t0 = Instant::now();
        let mut sum_f64 = 0.0f64;
        for i in 0..N-1 {
            let diff = points_f64[i] - points_f64[i + 1];
            let plane = diff.normalize();
            sum_f64 += plane.dot(points_f64[i]);
        }
        let combined_f64_time = t0.elapsed();
        black_box(sum_f64);

        println!("\nCombined (normalize + dot):");
        println!("  f32: {:>8.2} ms", combined_f32_time.as_secs_f64() * 1000.0);
        println!("  f64: {:>8.2} ms", combined_f64_time.as_secs_f64() * 1000.0);
        println!("  ratio: {:.2}x", combined_f64_time.as_secs_f64() / combined_f32_time.as_secs_f64());
    }
}
