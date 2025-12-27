use glam::Vec3;

use super::cell_builder::{CellFailure, F64CellBuilder, VertexData, VertexKey, MAX_PLANES};
use super::predicates::{PredKernel, PredResult, Sign};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyCertification {
    /// Keys are conservative: they may include ambiguous planes.
    Provisional,
    /// Keys contain only provably supporting planes (within the kernel’s certainty).
    Certified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CertifyFailure {
    /// A candidate generator appears to beat `g` at a purported vertex direction.
    /// This indicates missing constraints / insufficient neighbors (or bad termination).
    NeedMoreNeighbors,
    /// The predicate kernel cannot decide a sign.
    NeedMorePrecision,
}

#[inline]
fn invert_sign(sign: Sign) -> Sign {
    match sign {
        Sign::Neg => Sign::Pos,
        Sign::Zero => Sign::Zero,
        Sign::Pos => Sign::Neg,
    }
}

pub(super) fn certify_to_vertex_data_into(
    cell: &F64CellBuilder,
    kernel: &dyn PredKernel,
    support_data: &mut Vec<u32>,
    out: &mut Vec<VertexData>,
) -> Result<KeyCertification, CellFailure> {
    out.clear();
    if cell.is_failed() || cell.vertex_count() < 3 {
        return Ok(KeyCertification::Certified);
    }

    let gen_idx = cell.generator_index_u32();

    let mut support_tmp: Vec<u32> = Vec::with_capacity(MAX_PLANES + 1);
    let mut support_extra = [0u32; MAX_PLANES];
    let mut certification = KeyCertification::Certified;

    out.reserve(cell.vertex_count());

    for vi in 0..cell.vertex_count() {
        let v_pos = cell.vertex_pos_unit(vi);
        let pos = Vec3::new(v_pos.x as f32, v_pos.y as f32, v_pos.z as f32);

        let (plane_a, plane_b) = cell.vertex_def_planes(vi);
        let def_a = cell.plane_neighbor_index_u32(plane_a);
        let def_b = cell.plane_neighbor_index_u32(plane_b);

        let n_a = cell.plane_normal_unnorm(plane_a);
        let n_b = cell.plane_normal_unnorm(plane_b);

        // Orientation matters: cross(n_a, n_b) has two possible directions on S².
        // Match the actual stored vertex direction so determinant signs correspond to
        // dot(v_dir, n_c).
        let mut flip = false;
        let v_dir = n_a.cross(n_b);
        if v_dir.dot(v_pos) < 0.0 {
            flip = true;
        }

        let mut extra_len = 0usize;

        for pi in 0..cell.planes_count() {
            if pi == plane_a || pi == plane_b {
                continue;
            }

            let n_c = cell.plane_normal_unnorm(pi);
            match kernel.det3_sign(n_a, n_b, n_c) {
                PredResult::Certain(mut sign) => {
                    if flip {
                        sign = invert_sign(sign);
                    }
                    match sign {
                        Sign::Pos => {}
                        Sign::Zero => {
                            support_extra[extra_len] = cell.plane_neighbor_index_u32(pi);
                            extra_len += 1;
                        }
                        Sign::Neg => return Err(CellFailure::CertificationFailed),
                    }
                }
                PredResult::Uncertain => {
                    // Conservative: treat ambiguity as support, producing a stable but
                    // potentially non-minimal key.
                    certification = KeyCertification::Provisional;
                    support_extra[extra_len] = cell.plane_neighbor_index_u32(pi);
                    extra_len += 1;
                }
            }
        }

        let key = if extra_len == 0 {
            let mut a = gen_idx;
            let mut b = def_a;
            let mut c = def_b;
            F64CellBuilder::sort3_u32(&mut a, &mut b, &mut c);
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
                let len = u8::try_from(support_tmp.len()).map_err(|_| CellFailure::CertificationFailed)?;
                VertexKey::Support { start, len }
            } else {
                return Err(CellFailure::CertificationFailed);
            }
        };

        out.push((key, pos));
    }

    Ok(certification)
}
