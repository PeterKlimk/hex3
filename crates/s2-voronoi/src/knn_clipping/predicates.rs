use glam::DVec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Neg,
    Zero,
    Pos,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredResult {
    Certain(Sign),
    Uncertain,
}

pub trait PredKernel {
    fn det3_sign(&self, a: DVec3, b: DVec3, c: DVec3) -> PredResult;
}

#[inline]
fn det3_f64(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let (ax, ay, az) = (a.x, a.y, a.z);
    let (bx, by, bz) = (b.x, b.y, b.z);
    let (cx, cy, cz) = (c.x, c.y, c.z);
    ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
}

#[inline]
fn det3_err_bound(a: DVec3, b: DVec3, c: DVec3) -> f64 {
    // Conservative bound; can be tightened later.
    let aa = DVec3::new(a.x.abs(), a.y.abs(), a.z.abs());
    let ab = DVec3::new(b.x.abs(), b.y.abs(), b.z.abs());
    let ac = DVec3::new(c.x.abs(), c.y.abs(), c.z.abs());

    // |det(a,b,c)| <= sum_i |a_i| * (|b_j||c_k| + |b_k||c_j|)
    let m = aa.x * (ab.y * ac.z + ab.z * ac.y)
        + aa.y * (ab.x * ac.z + ab.z * ac.x)
        + aa.z * (ab.x * ac.y + ab.y * ac.x);

    // Safety multiplier for floating-point rounding in the expansion.
    let k = 64.0;
    k * f64::EPSILON * m
}

#[derive(Debug, Default, Clone, Copy)]
pub struct F64FilteredKernel;

impl PredKernel for F64FilteredKernel {
    #[inline]
    fn det3_sign(&self, a: DVec3, b: DVec3, c: DVec3) -> PredResult {
        let d = det3_f64(a, b, c);
        if !d.is_finite() {
            return PredResult::Uncertain;
        }
        if d == 0.0 {
            return PredResult::Certain(Sign::Zero);
        }

        let bound = det3_err_bound(a, b, c);
        if d > bound {
            PredResult::Certain(Sign::Pos)
        } else if d < -bound {
            PredResult::Certain(Sign::Neg)
        } else {
            PredResult::Uncertain
        }
    }
}
