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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredTier {
    FilteredF64,
    DoubleDouble,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KernelLadder {
    filtered: F64FilteredKernel,
    double_double: DoubleDoubleKernel,
}

impl KernelLadder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn tiers(&self) -> impl Iterator<Item = (PredTier, &dyn PredKernel)> + '_ {
        [
            (PredTier::FilteredF64, &self.filtered as &dyn PredKernel),
            (PredTier::DoubleDouble, &self.double_double as &dyn PredKernel),
        ]
        .into_iter()
    }
}

#[inline]
pub fn det3_f64(a: DVec3, b: DVec3, c: DVec3) -> f64 {
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

#[derive(Debug, Default, Clone, Copy)]
pub struct DoubleDoubleKernel;

#[derive(Clone, Copy, Debug)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

const SPLITTER: f64 = 134217729.0; // 2^27 + 1

#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let hi = c - (c - a);
    let lo = a - hi;
    (hi, lo)
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (a_hi, a_lo) = split(a);
    let (b_hi, b_lo) = split(b);
    let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, err)
}

#[inline]
fn dd_add(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    let (s, e) = two_sum(a.hi, b.hi);
    let e = e + a.lo + b.lo;
    let (hi, lo) = two_sum(s, e);
    DoubleDouble { hi, lo }
}

#[inline]
fn dd_sub(a: DoubleDouble, b: DoubleDouble) -> DoubleDouble {
    dd_add(
        a,
        DoubleDouble {
            hi: -b.hi,
            lo: -b.lo,
        },
    )
}

#[inline]
fn dd_from_prod(a: f64, b: f64) -> DoubleDouble {
    let (hi, lo) = two_prod(a, b);
    DoubleDouble { hi, lo }
}

#[inline]
fn dd_mul_f64(a: f64, b: DoubleDouble) -> DoubleDouble {
    let (p, e) = two_prod(a, b.hi);
    let e = e + a * b.lo;
    let (hi, lo) = two_sum(p, e);
    DoubleDouble { hi, lo }
}

#[inline]
fn det3_dd(a: DVec3, b: DVec3, c: DVec3) -> DoubleDouble {
    let m1 = dd_sub(dd_from_prod(b.y, c.z), dd_from_prod(b.z, c.y));
    let t1 = dd_mul_f64(a.x, m1);
    let m2 = dd_sub(dd_from_prod(b.x, c.z), dd_from_prod(b.z, c.x));
    let t2 = dd_mul_f64(a.y, m2);
    let m3 = dd_sub(dd_from_prod(b.x, c.y), dd_from_prod(b.y, c.x));
    let t3 = dd_mul_f64(a.z, m3);
    dd_add(dd_sub(t1, t2), t3)
}

impl PredKernel for DoubleDoubleKernel {
    #[inline]
    fn det3_sign(&self, a: DVec3, b: DVec3, c: DVec3) -> PredResult {
        let d = det3_dd(a, b, c);
        if !d.hi.is_finite() || !d.lo.is_finite() {
            return PredResult::Uncertain;
        }

        if d.hi > 0.0 {
            PredResult::Certain(Sign::Pos)
        } else if d.hi < 0.0 {
            PredResult::Certain(Sign::Neg)
        } else if d.lo > 0.0 {
            PredResult::Certain(Sign::Pos)
        } else if d.lo < 0.0 {
            PredResult::Certain(Sign::Neg)
        } else {
            PredResult::Certain(Sign::Zero)
        }
    }
}
