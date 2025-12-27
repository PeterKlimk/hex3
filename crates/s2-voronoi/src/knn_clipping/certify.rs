use glam::{DVec3, Vec3};

use super::cell_builder::{F64CellBuilder, VertexData, VertexKey, MAX_PLANES};
use super::predicates::{det3_f64, PredKernel, PredResult, Sign};

#[derive(Clone, Debug, PartialEq)]
pub enum CertifyError {
    /// The predicate kernel cannot decide a sign.
    NeedMorePrecision,
    /// Predicate sign is inconsistent with the already-clipped plane set.
    InvariantViolation(InvariantViolationInfo),
}

#[derive(Clone, Debug, PartialEq)]
pub struct InvariantViolationInfo {
    pub kind: ViolationKind,
    pub gen_idx: u32,
    pub vertex_idx: usize,
    pub def_a: u32,
    pub def_b: u32,
    pub n_a: DVec3,
    pub n_b: DVec3,
    pub v_pos: DVec3,
    /// For PlaneNeg: the violating plane
    pub violating_plane: Option<u32>,
    pub n_c: Option<DVec3>,
    /// Raw determinant values for debugging
    pub det_orientation: f64,
    pub det_plane: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViolationKind {
    /// det(n_a, n_b, v_pos) == 0: vertex lies in plane of its defining normals
    OrientationZero,
    /// det(n_a, n_b, n_c) has wrong sign: plane c should have clipped this vertex
    PlaneNeg,
    /// Support set has fewer than 3 elements after dedup
    SupportTooSmall,
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
) -> Result<(), CertifyError> {
    out.clear();
    if cell.is_failed() || cell.vertex_count() < 3 {
        return Ok(());
    }

    let gen_idx = cell.generator_index_u32();

    let support_base = support_data.len();
    let mut support_tmp: Vec<u32> = Vec::with_capacity(MAX_PLANES + 1);
    let mut support_extra = [0u32; MAX_PLANES];

    out.reserve(cell.vertex_count());

    for vi in 0..cell.vertex_count() {
        let v_pos = cell.vertex_pos_unit(vi);
        let pos = Vec3::new(v_pos.x as f32, v_pos.y as f32, v_pos.z as f32);

        let (plane_a, plane_b) = cell.vertex_def_planes(vi);
        let def_a = cell.plane_neighbor_index_u32(plane_a);
        let def_b = cell.plane_neighbor_index_u32(plane_b);

        let n_a = cell.plane_normal_unnorm(plane_a);
        let n_b = cell.plane_normal_unnorm(plane_b);

        let det_orientation = det3_f64(n_a, n_b, v_pos);

        // Orientation matters: cross(n_a, n_b) has two possible directions on S².
        // Match the actual stored vertex direction so determinant signs correspond to
        // dot(v_dir, n_c).
        let mut flip = false;
        match kernel.det3_sign(n_a, n_b, v_pos) {
            PredResult::Certain(Sign::Pos) => {}
            PredResult::Certain(Sign::Neg) => {
                flip = true;
            }
            PredResult::Certain(Sign::Zero) => {
                support_data.truncate(support_base);
                return Err(CertifyError::InvariantViolation(InvariantViolationInfo {
                    kind: ViolationKind::OrientationZero,
                    gen_idx,
                    vertex_idx: vi,
                    def_a,
                    def_b,
                    n_a,
                    n_b,
                    v_pos,
                    violating_plane: None,
                    n_c: None,
                    det_orientation,
                    det_plane: None,
                }));
            }
            PredResult::Uncertain => {
                support_data.truncate(support_base);
                return Err(CertifyError::NeedMorePrecision);
            }
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
                        Sign::Neg => {
                            let det_plane = det3_f64(n_a, n_b, n_c);
                            support_data.truncate(support_base);
                            return Err(CertifyError::InvariantViolation(InvariantViolationInfo {
                                kind: ViolationKind::PlaneNeg,
                                gen_idx,
                                vertex_idx: vi,
                                def_a,
                                def_b,
                                n_a,
                                n_b,
                                v_pos,
                                violating_plane: Some(cell.plane_neighbor_index_u32(pi)),
                                n_c: Some(n_c),
                                det_orientation,
                                det_plane: Some(det_plane),
                            }));
                        }
                    }
                }
                PredResult::Uncertain => {
                    support_data.truncate(support_base);
                    return Err(CertifyError::NeedMorePrecision);
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
                let len = match u8::try_from(support_tmp.len()) {
                    Ok(len) => len,
                    Err(_) => {
                        support_data.truncate(support_base);
                        return Err(CertifyError::InvariantViolation(InvariantViolationInfo {
                            kind: ViolationKind::SupportTooSmall,
                            gen_idx,
                            vertex_idx: vi,
                            def_a,
                            def_b,
                            n_a,
                            n_b,
                            v_pos,
                            violating_plane: None,
                            n_c: None,
                            det_orientation,
                            det_plane: None,
                        }));
                    }
                };
                VertexKey::Support { start, len }
            } else {
                support_data.truncate(support_base);
                return Err(CertifyError::InvariantViolation(InvariantViolationInfo {
                    kind: ViolationKind::SupportTooSmall,
                    gen_idx,
                    vertex_idx: vi,
                    def_a,
                    def_b,
                    n_a,
                    n_b,
                    v_pos,
                    violating_plane: None,
                    n_c: None,
                    det_orientation,
                    det_plane: None,
                }));
            }
        };

        out.push((key, pos));
    }

    Ok(())
}
